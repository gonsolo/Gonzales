"""Scene primitives, split out of geometry.mojo (the per-cluster module
split; see project_geometry_module_split memory). PrimId/Instance,
TriangleMesh/Ray/Intersection, Sphere+sphere_outward_normal, and
intersect_triangle/_alpha_hash/alpha_killed were four separate ranges in
geometry.mojo; every external dependency is the core
(Point3f/Vec3f/RGB/_is_real_ptr/cross/dot), confirmed by a symbol-reference
scan of each block with comments and docstrings stripped before moving it."""
from std.math import floor
from std.memory import bitcast
from .geometry import Point3f, Vec3f, RGB, _is_real_ptr, cross, dot

# ── Scene primitives ───────────────────────────────────────────────────────────

@fieldwise_init
struct PrimId(TrivialRegisterPassable):
    var id1: Int64
    var id2: Int64
    var materialIndex: Int64
    var instanceIdx: Int32   # -1 = ordinary top-level prim; else index into SceneDescriptor2_C.instances
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8

# ── Object instancing (two-level BVH: BLAS per template, TLAS instance leaves) ─

@fieldwise_init
struct Instance(TrivialRegisterPassable):
    """One placement of a template (BLAS). `objToWorld`/`worldToObj` are 16-float
    column-major matrices (same convention as transform.mojo). A TLAS leaf of
    PrimId.type == 6 has id1 = index into SceneDescriptor2_C.instances.
    `blasIdx` indexes SceneDescriptor2_C.blasNodesArr/blasPrimIdsArr (one
    private BVH2 per template, each a separate allocation — no shared-pool
    offset arithmetic needed). A BLAS's PrimId entries use ordinary type==0
    triangle encoding against the same global `meshes` array as the TLAS."""
    var objToWorld: SIMD[DType.float32, 16]
    var worldToObj: SIMD[DType.float32, 16]
    var blasIdx:    Int32


struct TriangleMesh(TrivialRegisterPassable):
    var points: Pointer[Float32, MutUntrackedOrigin]
    var faceIndices: Pointer[Int64, MutUntrackedOrigin]
    var vertexIndices: Pointer[Int64, MutUntrackedOrigin]
    var uvs: Pointer[Float32, MutUntrackedOrigin]   # nullable; stride 2 floats per vertex
    var normals: Pointer[Float32, MutUntrackedOrigin]  # nullable; stride 3 floats per vertex (shading normals)
    # pbrt `Shape "texture alpha"` / `"float alpha"` cut-out. `alpha` is an
    # alpha_w x alpha_h byte mask (linear alpha * 255, level 0 only -- pbrt
    # evaluates alpha at intersection time with zero filter width), shared
    # between every mesh that names the same file; dangling with alpha_w == 0
    # when the shape has no alpha texture, and then alpha_const (1 = opaque)
    # is the whole story. Read ONLY by alpha_killed below. vulkanrt.h mirrors
    # this layout field for field.
    var alpha: Pointer[UInt8, MutUntrackedOrigin]
    var alpha_w: Int32
    var alpha_h: Int32
    var alpha_const: Float32
    var _alpha_pad: Int32

    def __init__(
        out self,
        points: Pointer[Float32, MutUntrackedOrigin],
        faceIndices: Pointer[Int64, MutUntrackedOrigin],
        vertexIndices: Pointer[Int64, MutUntrackedOrigin],
        uvs: Pointer[Float32, MutUntrackedOrigin],
        normals: Pointer[Float32, MutUntrackedOrigin],
        alpha: Pointer[UInt8, MutUntrackedOrigin] = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
        alpha_w: Int32 = 0,
        alpha_h: Int32 = 0,
        alpha_const: Float32 = 1.0,
    ):
        self.points = points
        self.faceIndices = faceIndices
        self.vertexIndices = vertexIndices
        self.uvs = uvs
        self.normals = normals
        self.alpha = alpha
        self.alpha_w = alpha_w
        self.alpha_h = alpha_h
        self.alpha_const = alpha_const
        self._alpha_pad = 0

# ── Ray ───────────────────────────────────────────────────────────────────────
# See: docs/03_shapes_and_acceleration.md

@fieldwise_init
struct Ray(TrivialRegisterPassable):
    """A ray: a world-space origin point and a unit direction vector."""
    var origin: Point3f
# <<listing: Ray>>
    var direction: Vec3f

# ── Intersection ──────────────────────────────────────────────────────────────
# <</listing>>

@fieldwise_init
struct Intersection(TrivialRegisterPassable):
    var primId: PrimId
    var tHit: Float32
    var u: Float32
    var v: Float32
    var hit: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8


@fieldwise_init
struct Sphere(TrivialRegisterPassable):
    """Analytical sphere primitive. Exact intersection, exact normals.
    isAreaLight == 1 → sphere emits light (NEE via solid-angle cone sampling).
    """
    var center: Point3f
    var radius: Float32
    var materialIndex: Int32
    var isAreaLight: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var emission: RGB

@always_inline
def sphere_outward_normal(hit: Point3f, center: Point3f) -> Vec3f:
    """Normalized outward normal at a point on an analytic sphere's surface
    (normalize(hit - center)). Degenerate (hit == center) returns the zero
    vector, same fallback as every inline call site this replaces."""
    var n = hit - center
    var nl = n.length()
    return (n / nl) if nl > Float32(0.0) else n


@always_inline
def intersect_triangle(
    ray_org: Vec3f,
    ray_dir: Vec3f,
    p0: Vec3f,
    p1: Vec3f,
    p2: Vec3f,
    tMax: Float32
) -> Tuple[Bool, Float32, Float32, Float32]:
    """Möller–Trumbore ray-triangle intersection.
    Returns (hit, t, u, v) where (u,v) are barycentric coordinates.
    See: docs/03_shapes_and_acceleration.md — Triangle Intersection.
    """
    var e1 = p1 - p0
    var e2 = p2 - p0
    var pvec = cross(ray_dir, e2)
    var det = dot(e1, pvec)

    if det > -0.0000001 and det < 0.0000001:
        return (False, tMax, 0.0, 0.0)

    var invDet = 1.0 / det
    var tvec = ray_org - p0
    var u = dot(tvec, pvec) * invDet

    # Watertight edge tolerance: Moller-Trumbore's barycentric tests are not
    # guaranteed watertight at a shared edge between two adjacent triangles
    # under floating-point rounding -- a ray passing extremely close to a
    # shared edge/vertex can compute u/v/u+v just outside [0,1] for BOTH
    # triangles independently (each rejects, neither accepts), producing a
    # real miss (hit=0) despite the ray genuinely crossing the mesh's
    # surface. Reproduced concretely: cornell-box's back wall and floor
    # (each a quad split into 2 triangles via "integer indices [0 1 2 0 2
    # 3]") showed a diagonal line of missed-intersection pixels exactly
    # along that shared diagonal, visible even in the raw albedo AOV (pure
    # geometry/material lookup, no lighting math involved at all) --
    # confirming this was a primary-ray intersection gap, not a shading
    # bug. A small epsilon tolerance on the edge tests (growing each
    # triangle by a negligible sliver in barycentric space) is the standard
    # pragmatic fix for this class of crack, short of a full watertight
    # (Woop et al. 2013) rewrite of the intersection algorithm.
    comptime _EDGE_EPS = Float32(1e-5)
    if u < -_EDGE_EPS or u > 1.0 + _EDGE_EPS:
        return (False, tMax, 0.0, 0.0)

    var qvec = cross(tvec, e1)
    var v = dot(ray_dir, qvec) * invDet

    if v < -_EDGE_EPS or u + v > 1.0 + _EDGE_EPS:
        return (False, tMax, 0.0, 0.0)

    var t = dot(e2, qvec) * invDet
    if t <= 0.0 or t > tMax:
        return (False, tMax, 0.0, 0.0)

    return (True, t, u, v)


@always_inline
def _alpha_hash(ray_org: Vec3f, ray_dir: Vec3f, key: Int) -> Float32:
    """Uniform [0,1) from the ray and the triangle: pbrt's HashFloat(o, d)
    role. Salting with the triangle makes each cut-out layer an independent
    coin, as pbrt's CPU build gets by re-hashing the respawned ray."""
    var h = UInt64(key) * UInt64(0x9E3779B97F4A7C15)
    var o = SIMD[DType.float32, 4](ray_org.x, ray_org.y, ray_org.z, Float32(0))
    var d = SIMD[DType.float32, 4](ray_dir.x, ray_dir.y, ray_dir.z, Float32(0))
    var ob = bitcast[DType.uint32, 4](o)
    var db = bitcast[DType.uint32, 4](d)
    comptime for k in range(3):
        h = (h ^ UInt64(ob[k])) * UInt64(0xBF58476D1CE4E5B9)
        h ^= h >> 29
        h = (h ^ UInt64(db[k])) * UInt64(0x94D049BB133111EB)
        h ^= h >> 32
    h ^= h >> 31
    return Float32(UInt32(h >> 40)) * Float32(1.0 / 16777216.0)


@always_inline
def alpha_killed(mesh: TriangleMesh, v0: Int, v1: Int, v2: Int, bu: Float32, bv: Float32,
                 ray_org: Vec3f, ray_dir: Vec3f, key: Int) -> Bool:
    """True when a hit at barycentrics (bu, bv) on this triangle is cut out by
    the shape's alpha and traversal must look past it. pbrt-v4's any-hit rule
    (gpu/optix/optix.cu alphaKilled, cpu/primitive.cpp): alpha >= 1 keeps the
    hit, alpha <= 0 drops it, anything between drops it with probability
    1 - alpha. Every triangle-hit site in bvh.mojo calls this, so camera,
    NEE-shadow, photon and light-subpath rays of all three integrators agree
    about which leaves exist.

    The mask is sampled bilinearly at level 0 in pbrt's convention: V flipped,
    repeat wrap, texel centres at half-integers. A mesh without UVs gets
    pbrt's default per-triangle (0,0) (1,0) (1,1)."""
    if mesh.alpha_w == Int32(0):
        if mesh.alpha_const >= Float32(1.0):
            return False
        if mesh.alpha_const <= Float32(0.0):
            return True
        return _alpha_hash(ray_org, ray_dir, key) > mesh.alpha_const
    var w0 = Float32(1.0) - bu - bv
    var su: Float32
    var tv: Float32
    if _is_real_ptr(mesh.uvs):
        su = w0*mesh.uvs[unsafe_offset=v0*2]   + bu*mesh.uvs[unsafe_offset=v1*2]   + bv*mesh.uvs[unsafe_offset=v2*2]
        tv = w0*mesh.uvs[unsafe_offset=v0*2+1] + bu*mesh.uvs[unsafe_offset=v1*2+1] + bv*mesh.uvs[unsafe_offset=v2*2+1]
    else:
        su = bu + bv
        tv = bv
    tv = Float32(1.0) - tv
    var w = Int(mesh.alpha_w)
    var h = Int(mesh.alpha_h)
    var x = su * Float32(w) - Float32(0.5)
    var y = tv * Float32(h) - Float32(0.5)
    var xf = floor(x)
    var yf = floor(y)
    var dx = x - xf
    var dy = y - yf
    var x0 = Int(xf) % w
    if x0 < 0: x0 += w
    var y0 = Int(yf) % h
    if y0 < 0: y0 += h
    var x1 = x0 + 1
    if x1 == w: x1 = 0
    var y1 = y0 + 1
    if y1 == h: y1 = 0
    var a00 = Float32(mesh.alpha[unsafe_offset=y0*w + x0])
    var a10 = Float32(mesh.alpha[unsafe_offset=y0*w + x1])
    var a01 = Float32(mesh.alpha[unsafe_offset=y1*w + x0])
    var a11 = Float32(mesh.alpha[unsafe_offset=y1*w + x1])
    var a = ((a00 * (Float32(1.0) - dx) + a10 * dx) * (Float32(1.0) - dy)
             + (a01 * (Float32(1.0) - dx) + a11 * dx) * dy) * Float32(1.0 / 255.0)
    if a >= Float32(1.0):
        return False
    if a <= Float32(0.0):
        return True
    return _alpha_hash(ray_org, ray_dir, key) > a



