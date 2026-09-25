"""Lights, split out of geometry.mojo (the per-cluster module split; see
project_geometry_module_split memory). AreaLight_C, DistantLight_C/
PointLight_C/InfiniteLight_C and LightSampler_C were three physically
separate ranges in geometry.mojo (the last two sitting inside what had grown
into the media section, not their own) -- each depends only on the core
(Point3f/Vec3f/RGB/_is_real_ptr), confirmed by scanning each block with
comments and docstrings stripped before moving it."""
from std.math import max, min
from .geometry import Point3f, Vec3f, RGB, _is_real_ptr

# ── Lights ────────────────────────────────────────────────────────────────────
# See: docs/06_lights_and_materials.md

struct AreaLight_C(TrivialRegisterPassable):
    """A sampleable area light. kind==0: a triangle mesh (meshIdx indexes
    TriangleMesh_C, n_tris triangles, total_area = mesh surface area).
    kind==1: a native curve (meshIdx reused as the curve's index into the
    scene's Curve_C array; n_tris unused; total_area = the curve's tube
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
def area_light_pick_triangle(al: AreaLight_C, u: Float32) -> Int:
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
struct DistantLight_C(TrivialRegisterPassable):
    """A directional (infinite-distance) light.
    `direction` points FROM the light TOWARD the scene (world space).
    """
    var direction: Vec3f
    var _pad: Float32
    var emission: RGB
    var _pad2: Float32

@fieldwise_init
struct PointLight_C(TrivialRegisterPassable):
    """An isotropic point light at a world-space position."""
    var position: Point3f
    var _pad: Float32
    var intensity: RGB
    var _pad2: Float32

@fieldwise_init
struct InfiniteLight_C(TrivialRegisterPassable):
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


@fieldwise_init
struct LightSampler_C(TrivialRegisterPassable):
    """Power-weighted CDF over area lights.
    cdf[0]=0, cdf[n]=1; pdf[i] = cdf[i+1] - cdf[i] = power_i / total_power.
    Built at parse time; on GPU the cdf pointer is patched to device memory.
    """
    var cdf: Pointer[Float32, MutUntrackedOrigin]  # n+1 entries
    var n: Int32
    var _pad: Int32

@always_inline
def light_sampler_sample(ls: LightSampler_C, u: Float32) -> Tuple[Int, Float32]:
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
def light_sampler_pdf(ls: LightSampler_C, light_idx: Int32) -> Float32:
    """Selection pdf for a KNOWN light index -- the inverse of
    light_sampler_sample's (index, pdf) draw, needed to re-evaluate a
    specific light's pdf without redrawing it (e.g. ReSTIR DI's MIS weight
    for an already-chosen reservoir winner, restir_di.mojo)."""
    var i = Int(light_idx)
    return max(ls.cdf[unsafe_offset=i + 1] - ls.cdf[unsafe_offset=i], Float32(1e-6))


