"""Lights, split out of geometry.mojo (the per-cluster module split; see
project_geometry_module_split memory). AreaLight, DistantLight/
PointLight/InfiniteLight and LightSampler were three physically
separate ranges in geometry.mojo (the last two sitting inside what had grown
into the media section, not their own) -- each depends only on the core
(Point3f/Vec3f/RGB/_is_real_ptr), confirmed by scanning each block with
comments and docstrings stripped before moving it."""
from std.memory.alloc import unsafe_alloc
from std.math import max, min, sqrt
from .geometry import Point3f, Vec3f, RGB, _is_real_ptr
from .portal_light import portal_frame, portal_ray_crosses

# ── Lights ────────────────────────────────────────────────────────────────────
# See: docs/06_lights_and_materials.md

struct AreaLight(TrivialRegisterPassable):
    """A sampleable area light. kind==0: a triangle mesh (meshIdx indexes
    TriangleMesh, n_tris triangles, total_area = mesh surface area).
    kind==1: a native curve (meshIdx reused as the curve's index into the
    scene's Curve array; n_tris unused; total_area = the curve's tube
    lateral surface area, see curve_light_tube_area). A mesh light's triangle
    is picked area-weighted (area_light_pick_triangle); a curve light's piece
    is still picked uniformly -- see sample_area_light_uniform (sppm.mojo)."""
    var meshIdx: Int32
    var n_tris: Int32       # number of triangles in this light mesh (kind==0 only)
    var emission: RGB
    var total_area: Float32 # total surface area of this light mesh or curve tube
    var kind: Int8          # 0 = mesh triangle light, 1 = curve light
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    # Cumulative triangle areas / total_area, n_tris entries ending at 1:
    # area_light_pick_triangle draws a triangle in PROPORTION TO ITS AREA, so
    # a uniform point on it has density exactly 1 / total_area -- the density
    # every consumer (PT NEE, VCM light paths and MIS, SPPM photons) divides
    # by. Dangling (curves, hand-built fixtures) falls back to a uniform pick,
    # which is only right when the triangles are equal.
    var tri_cdf: Pointer[Float32, MutUntrackedOrigin]

    def __init__(out self, meshIdx: Int32, n_tris: Int32, emission: RGB, total_area: Float32,
                 kind: Int8, _pad0: Int8, _pad1: Int8, _pad2: Int8,
                 tri_cdf: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()):
        self.meshIdx = meshIdx
        self.n_tris = n_tris
        self.emission = emission
        self.total_area = total_area
        self.kind = kind
        self._pad0 = _pad0
        self._pad1 = _pad1
        self._pad2 = _pad2
        self.tri_cdf = tri_cdf


@always_inline
def area_light_pick_triangle(al: AreaLight, u: Float32) -> Int:
    """Triangle index of mesh light `al` for a uniform u in [0,1), picked in
    proportion to triangle area (binary search of al.tri_cdf).

    Picking uniformly BY INDEX instead -- as every call site did until
    2026-09-24 -- gives a point on triangle t the density 1/(n_tris * A_t),
    while every weight and flux assumed 1/total_area: small triangles
    over-emitted per unit area. On barcelona-pavilion-night's candle flame
    (1984 triangles, areas spread 9x) the light-side VCM strategies then
    disagreed with the camera side in expectation, which is what made MIS
    weight changes move the mean at the lanterns."""
    var n = Int(max(Int(al.n_tris), 1))
    if not _is_real_ptr(al.tri_cdf):
        return min(Int(u * Float32(n)), n - 1)
    var lo = 0
    var hi = n - 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if al.tri_cdf[unsafe_offset=mid] > u:
            hi = mid
        else:
            lo = mid + 1
    return lo


@fieldwise_init
struct DistantLight(TrivialRegisterPassable):
    """A directional (infinite-distance) light.
    `direction` points FROM the light TOWARD the scene (world space).
    """
    var direction: Vec3f
    var _pad: Float32
    var emission: RGB
    var _pad2: Float32

@fieldwise_init
struct PointLight(TrivialRegisterPassable):
    """An isotropic point light at a world-space position."""
    var position: Point3f
    var _pad: Float32
    var intensity: RGB
    var _pad2: Float32

@fieldwise_init
struct InfiniteLight(TrivialRegisterPassable):
    """An environment map (lat-long HDRI), importance-sampled via 2D CDF.
    See: docs/06_lights_and_materials.md — Infinite Area Lights.
    """
    var scale: RGB
    var tex_idx: Int32   # -1 = solid colour, >= 0 = texture
    var cdf_w: Int32     # env-map pixel width (also CDF width; 0 = no texture)
    var cdf_h: Int32     # env-map pixel height
    var cdf_ptr: Pointer[Float32, MutUntrackedOrigin]   # flat 2D CDF (marginal + conditional)
    var pixels_ptr: Pointer[Float32, MutUntrackedOrigin] # raw HDR pixels, 3 floats/pixel (CPU only)
    var world_to_light: Pointer[Float32, MutUntrackedOrigin]  # 16-float col-major inverse of light CTM
    # pbrt-v4 portal (`LightSource "infinite" "point3 portal" [...]`): restricts
    # this light to the solid angle actually visible through a world-space
    # quadrilateral (p0,p1,p2,p3, wound in order) instead of the whole sphere --
    # see portal_light.mojo. has_portal==0 (the common case) is the ordinary,
    # unrestricted environment light this struct always was.
    var has_portal: Int32
    var portal_p0: Point3f
    var portal_p1: Point3f
    var portal_p2: Point3f
    var portal_p3: Point3f
    # 1: pixels_ptr holds Float16 triples (the GPU copy of a big map); the host keeps Float32.
    var pixels_half: Int32

    @always_inline
    def portal_admits(self, origin: Point3f, dir: Vec3f) -> Bool:
        """pbrt's portal Le: a light with a portal is seen only by rays through the portal quad."""
        if self.has_portal == Int32(0):
            return True
        var (e1, e2, n, _) = portal_frame(self.portal_p0, self.portal_p1, self.portal_p3)
        return portal_ray_crosses(origin, dir, self.portal_p0, e1, e2, n)

# Flat 2D sampling table of an equal-area environment map: (ih+1) marginal entries, then ih rows of (iw+1) conditional entries.
def env_build_cdf(pixels: Pointer[Float32, MutUntrackedOrigin], iw: Int, ih: Int) -> Pointer[Float32, MutUntrackedOrigin]:
    var cdf_size = (ih + 1) + ih * (iw + 1)
    var cdf_buf = unsafe_alloc[Float32](cdf_size)
    var row_sums = unsafe_alloc[Float32](ih)
    for ry in range(ih):
        var row_sum = Float32(0.0)
        for rx in range(iw):
            var r2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 0]
            var g2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 1]
            var b2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 2]
            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
            row_sum += lum
        row_sums[unsafe_offset=ry] = row_sum
    cdf_buf[unsafe_offset=0] = Float32(0.0)
    for ry in range(ih):
        cdf_buf[unsafe_offset=ry + 1] = cdf_buf[unsafe_offset=ry] + row_sums[unsafe_offset=ry]
    var total = cdf_buf[unsafe_offset=ih]
    if total > Float32(0.0):
        var inv_total = Float32(1.0) / total
        for ry in range(ih + 1):
            cdf_buf[unsafe_offset=ry] *= inv_total
    for ry in range(ih):
        var base = (ih + 1) + ry * (iw + 1)
        cdf_buf[unsafe_offset=base] = Float32(0.0)
        for rx in range(iw):
            var r2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 0]
            var g2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 1]
            var b2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 2]
            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
            cdf_buf[unsafe_offset=base + rx + 1] = cdf_buf[unsafe_offset=base + rx] + lum
        var row_total = cdf_buf[unsafe_offset=base + iw]
        if row_total > Float32(0.0):
            var inv_rt = Float32(1.0) / row_total
            for rx in range(iw + 1):
                cdf_buf[unsafe_offset=base + rx] *= inv_rt
    row_sums.unsafe_free()
    return cdf_buf


# 2x2 box average of an RGB map (odd edges dropped).
def env_downsample(pixels: Pointer[Float32, MutUntrackedOrigin], iw: Int, ih: Int) -> Pointer[Float32, MutUntrackedOrigin]:
    var ow = iw // 2
    var oh = ih // 2
    var out = unsafe_alloc[Float32](ow * oh * 3)
    for y in range(oh):
        for x in range(ow):
            for c in range(3):
                var t = pixels[unsafe_offset=((2 * y) * iw + 2 * x) * 3 + c] + pixels[unsafe_offset=((2 * y) * iw + 2 * x + 1) * 3 + c]
                t += pixels[unsafe_offset=((2 * y + 1) * iw + 2 * x) * 3 + c] + pixels[unsafe_offset=((2 * y + 1) * iw + 2 * x + 1) * 3 + c]
                out[unsafe_offset=(y * ow + x) * 3 + c] = t * Float32(0.25)
    return out


@always_inline
def il_rgb(il: InfiniteLight, texel: Int) -> RGB:
    if il.pixels_half != Int32(0):
        var h = il.pixels_ptr.unsafe_bitcast[Float16]()
        return RGB(Float32(h[unsafe_offset=texel * 3]), Float32(h[unsafe_offset=texel * 3 + 1]), Float32(h[unsafe_offset=texel * 3 + 2]))
    return RGB(il.pixels_ptr[unsafe_offset=texel * 3], il.pixels_ptr[unsafe_offset=texel * 3 + 1], il.pixels_ptr[unsafe_offset=texel * 3 + 2])


comptime LIGHT_BOUNDS_STRIDE: Int = 12
comptime LIGHT_IMPORTANCE_MAX: Int = 32


@fieldwise_init
struct LightSampler(TrivialRegisterPassable):
    """Area-light picker. cdf[0]=0, cdf[n]=1; pdf[i] = cdf[i+1] - cdf[i] (the power pick, mixed with uniform).
    With has_bounds != 0 (at most LIGHT_IMPORTANCE_MAX lights, all meshes) the buffer continues after the n+1 CDF entries with
    LIGHT_BOUNDS_STRIDE floats per light (box min, box max, emission axis, power, cos of the normal cone, two-sided), and the
    pick at a point follows pbrt's light importance instead. Built at parse time; on GPU the pointer is patched to device memory.
    """
    var cdf: Pointer[Float32, MutUntrackedOrigin]
    var n: Int32
    var has_bounds: Int32


@always_inline
def _safe_sqrt(x: Float32) -> Float32:
    return sqrt(max(x, Float32(0.0)))


@always_inline
def _cos_sub_clamped(sin_a: Float32, cos_a: Float32, sin_b: Float32, cos_b: Float32) -> Float32:
    return Float32(1.0) if cos_a > cos_b else cos_a * cos_b + sin_a * sin_b


@always_inline
def _sin_sub_clamped(sin_a: Float32, cos_a: Float32, sin_b: Float32, cos_b: Float32) -> Float32:
    return Float32(0.0) if cos_a > cos_b else sin_a * cos_b - cos_a * sin_b


@always_inline
def light_importance(ls: LightSampler, i: Int, p: Vec3f) -> Float32:
    """pbrt's LightBounds::Importance for a diffuse emitter (cos theta_e = 0), without the receiver-normal factor."""
    var b = ls.cdf.unsafe_offset(Int(ls.n) + 1 + i * LIGHT_BOUNDS_STRIDE)
    var cx = Float32(0.5) * (b[unsafe_offset=0] + b[unsafe_offset=3]); var cy = Float32(0.5) * (b[unsafe_offset=1] + b[unsafe_offset=4]); var cz = Float32(0.5) * (b[unsafe_offset=2] + b[unsafe_offset=5])
    var dx = p.x - cx; var dy = p.y - cy; var dz = p.z - cz
    var dist2 = dx * dx + dy * dy + dz * dz
    var ex = b[unsafe_offset=3] - b[unsafe_offset=0]; var ey = b[unsafe_offset=4] - b[unsafe_offset=1]; var ez = b[unsafe_offset=5] - b[unsafe_offset=2]
    var diag = sqrt(ex * ex + ey * ey + ez * ez)
    var d2 = max(dist2, Float32(0.5) * diag)
    var inv = Float32(1.0) / sqrt(max(dist2, Float32(1e-20)))
    var cos_w = (b[unsafe_offset=6] * dx + b[unsafe_offset=7] * dy + b[unsafe_offset=8] * dz) * inv
    if b[unsafe_offset=11] != Float32(0.0):
        cos_w = abs(cos_w)
    var sin_w = _safe_sqrt(Float32(1.0) - cos_w * cos_w)
    var r2 = Float32(0.25) * diag * diag
    var inside = p.x >= b[unsafe_offset=0] and p.x <= b[unsafe_offset=3] and p.y >= b[unsafe_offset=1] and p.y <= b[unsafe_offset=4] and p.z >= b[unsafe_offset=2] and p.z <= b[unsafe_offset=5]
    var cos_b = Float32(-1.0)
    if not inside and dist2 >= r2:
        cos_b = _safe_sqrt(Float32(1.0) - r2 / dist2)
    var sin_b = _safe_sqrt(Float32(1.0) - cos_b * cos_b)
    var cos_o = b[unsafe_offset=10]
    var sin_o = _safe_sqrt(Float32(1.0) - cos_o * cos_o)
    var cos_x = _cos_sub_clamped(sin_w, cos_w, sin_o, cos_o)
    var sin_x = _sin_sub_clamped(sin_w, cos_w, sin_o, cos_o)
    var cos_p = _cos_sub_clamped(sin_x, cos_x, sin_b, cos_b)
    if cos_p <= Float32(0.0):
        return Float32(0.0)
    return max(b[unsafe_offset=9] * cos_p / d2, Float32(0.0))


@always_inline
def light_sampler_sample_at(ls: LightSampler, p: Vec3f, u: Float32) -> Tuple[Int, Float32]:
    """(light_index, selection_pdf) for a shading point p; (-1, 0) when no light can reach it."""
    if ls.has_bounds == Int32(0):
        return light_sampler_sample(ls, u)
    var total = Float32(0.0)
    for i in range(Int(ls.n)):
        total += light_importance(ls, i, p)
    if total <= Float32(0.0):
        return (-1, Float32(0.0))
    var target = u * total
    var acc = Float32(0.0)
    var last = -1
    for i in range(Int(ls.n)):
        var w = light_importance(ls, i, p)
        if w > Float32(0.0):
            last = i
            acc += w
            if acc > target:
                return (i, max(w / total, Float32(1e-6)))
    return (last, max(light_importance(ls, last, p) / total, Float32(1e-6)))


@always_inline
def light_sampler_pdf_at(ls: LightSampler, light_idx: Int32, p: Vec3f) -> Float32:
    """The selection pdf light_sampler_sample_at gives light_idx at p."""
    if ls.has_bounds == Int32(0):
        return light_sampler_pdf(ls, light_idx)
    var total = Float32(0.0)
    for i in range(Int(ls.n)):
        total += light_importance(ls, i, p)
    if total <= Float32(0.0):
        return Float32(0.0)
    return light_importance(ls, Int(light_idx), p) / total


@always_inline
def light_sampler_sample(ls: LightSampler, u: Float32) -> Tuple[Int, Float32]:
    """Binary-search the CDF. Returns (light_index, selection_pdf)."""
    var lo = 0
    var hi = Int(ls.n) - 1
    while lo < hi:
        var mid = (lo + hi) >> 1
        if ls.cdf[unsafe_offset=mid + 1] <= u:
            lo = mid + 1
        else:
            hi = mid
    var pdf = ls.cdf[unsafe_offset=lo + 1] - ls.cdf[unsafe_offset=lo]
    return (lo, max(pdf, Float32(1e-6)))

@always_inline
def light_sampler_pdf(ls: LightSampler, light_idx: Int32) -> Float32:
    """Selection pdf for a KNOWN light index -- the inverse of
    light_sampler_sample's (index, pdf) draw, needed to re-evaluate a
    specific light's pdf without redrawing it (e.g. ReSTIR DI's MIS weight
    for an already-chosen reservoir winner, restir_di.mojo)."""
    var i = Int(light_idx)
    return max(ls.cdf[unsafe_offset=i + 1] - ls.cdf[unsafe_offset=i], Float32(1e-6))


