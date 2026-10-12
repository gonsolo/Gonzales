"""Scene primitives, split out of geometry.mojo (the per-cluster module
split; see project_geometry_module_split memory). PrimId/Instance,
TriangleMesh/Ray/Intersection, Sphere+sphere_outward_normal, and
intersect_triangle/_alpha_hash/alpha_killed were four separate ranges in
geometry.mojo; every external dependency is the core
(Point3f/Vec3f/RGB/_is_real_ptr/cross/dot), confirmed by a symbol-reference
scan of each block with comments and docstrings stripped before moving it."""
from std.math import floor, sqrt, abs, fma
from std.memory import bitcast
from .geometry import Point3f, Vec3f, RGB, _is_real_ptr, cross, dot

# ── Scene primitives ───────────────────────────────────────────────────────────

@fieldwise_init
struct PrimId(TrivialRegisterPassable):
    var id1: Int64
    var id2: Int64
    var materialIndex: Int64
    var instanceIdx: Int32   # -1 = ordinary top-level prim; else index into SceneView.instances
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8

# ── Object instancing (two-level BVH: BLAS per template, TLAS instance leaves) ─

struct Instance(TrivialRegisterPassable):
    """One placement of a template (BLAS): 64 bytes, the layout of an RT-core TLAS instance record (docs/rtcore), so
    that a pruned RT scene shades straight from its TLAS. Holds the world-to-object transform as the top three rows of
    a row-major matrix (it is affine, so the fourth row is 0 0 0 1); the object-to-world one is inverted from it where
    needed, which is once a shaded hit and never during traversal. cNr is row r of column n.
    A TLAS leaf of PrimId.type == 6 has id1 = index into SceneView.instances.
    `blasIdx` indexes SceneView.blasNodesArr/blasPrimIdsArr. The words `id`, `ptr0` and `ptr1` belong to the RT cores."""
    var id: Int32
    var blasIdx: Int32
    var ptr0: Int32
    var ptr1: Int32
    var c0x: Float32
    var c1x: Float32
    var c2x: Float32
    var c3x: Float32
    var c0y: Float32
    var c1y: Float32
    var c2y: Float32
    var c3y: Float32
    var c0z: Float32
    var c1z: Float32
    var c2z: Float32
    var c3z: Float32

    def __init__(out self, worldToObj: SIMD[DType.float32, 16], blasIdx: Int32):
        self.id = Int32(0); self.blasIdx = blasIdx; self.ptr0 = Int32(0); self.ptr1 = Int32(0)
        self.c0x = worldToObj[0]; self.c0y = worldToObj[1]; self.c0z = worldToObj[2]
        self.c1x = worldToObj[4]; self.c1y = worldToObj[5]; self.c1z = worldToObj[6]
        self.c2x = worldToObj[8]; self.c2y = worldToObj[9]; self.c2z = worldToObj[10]
        self.c3x = worldToObj[12]; self.c3y = worldToObj[13]; self.c3z = worldToObj[14]

    @always_inline
    def world_to_obj(self) -> SIMD[DType.float32, 16]:
        return SIMD[DType.float32, 16](
            self.c0x, self.c0y, self.c0z, Float32(0), self.c1x, self.c1y, self.c1z, Float32(0),
            self.c2x, self.c2y, self.c2z, Float32(0), self.c3x, self.c3y, self.c3z, Float32(1))

    @always_inline
    def obj_to_world(self) -> SIMD[DType.float32, 16]:
        # Inverse of an affine map: invert the 3x3 by cofactors, then carry the translation through it.
        var a = self.c0x; var b = self.c1x; var c = self.c2x
        var d = self.c0y; var e = self.c1y; var f = self.c2y
        var g = self.c0z; var h = self.c1z; var i = self.c2z
        var co00 = e * i - f * h; var co01 = f * g - d * i; var co02 = d * h - e * g
        var inv_det = Float32(1) / (a * co00 + b * co01 + c * co02)
        var r00 = co00 * inv_det; var r01 = (c * h - b * i) * inv_det; var r02 = (b * f - c * e) * inv_det
        var r10 = co01 * inv_det; var r11 = (a * i - c * g) * inv_det; var r12 = (c * d - a * f) * inv_det
        var r20 = co02 * inv_det; var r21 = (b * g - a * h) * inv_det; var r22 = (a * e - b * d) * inv_det
        var tx = -(r00 * self.c3x + r01 * self.c3y + r02 * self.c3z)
        var ty = -(r10 * self.c3x + r11 * self.c3y + r12 * self.c3z)
        var tz = -(r20 * self.c3x + r21 * self.c3y + r22 * self.c3z)
        return SIMD[DType.float32, 16](
            r00, r10, r20, Float32(0), r01, r11, r21, Float32(0), r02, r12, r22, Float32(0), tx, ty, tz, Float32(1))


struct TriangleMesh(TrivialRegisterPassable):
    var points: Pointer[Float32, MutUntrackedOrigin]
    var faceIndices: Pointer[Int32, MutUntrackedOrigin]
    var vertexIndices: Pointer[Int32, MutUntrackedOrigin]
    var uvs: Pointer[Float32, MutUntrackedOrigin]   # nullable; stride 2 floats per vertex
    # Shading normals, nullable: one 32-bit word a vertex, an octahedral encoding (encode_normal; read with
    # mesh_normal). The pointer is typed Float32 only so that a mesh without normals constructs as before.
    var normals: Pointer[Float32, MutUntrackedOrigin]
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
    var emit_two_sided: Int32   # area-light mesh emits from both faces (pbrt twosided)
    # Material per triangle, for a merged object template (PrimId.type 7, see bvh._traverse_blas_triangles);
    # dangling for every other mesh, whose material is in its PrimIds.
    var materials: Pointer[Int32, MutUntrackedOrigin]
    # 1 = a soup of quads whose texture coordinates are each face's own unit square: vertex v is corner v mod 4
    # of (0,0) (1,0) (1,1) (0,1). They are then not stored at all (`uvs` is dangling); read them with mesh_uv.
    var uv_corners: Int32
    # 1 = a soup of quads stored as such: quad q has vertices 4q..4q+3 and is the two triangles 2q = (0, 1, 2)
    # and 2q+1 = (0, 2, 3) of them. There is then no vertexIndices array, and faceIndices and materials hold
    # one entry a quad. Read through mesh_vi, mesh_face and mesh_material.
    var quads: Int32

    def __init__(
        out self,
        points: Pointer[Float32, MutUntrackedOrigin],
        faceIndices: Pointer[Int32, MutUntrackedOrigin],
        vertexIndices: Pointer[Int32, MutUntrackedOrigin],
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
        self.materials = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.uv_corners = Int32(0)
        self.quads = Int32(0)
        self.emit_two_sided = 0

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
    return intersect_triangle_edges(ray_org, ray_dir, p0, p1 - p0, p2 - p0, tMax)


@always_inline
def intersect_triangle_edges(
    ray_org: Vec3f,
    ray_dir: Vec3f,
    p0: Vec3f,
    e1: Vec3f,
    e2: Vec3f,
    tMax: Float32
) -> Tuple[Bool, Float32, Float32, Float32]:
    """intersect_triangle with the edges p1 - p0 and p2 - p0 precomputed."""
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
def mesh_vi(mesh: TriangleMesh, flat: Int) -> Int:
    """Vertex of a triangle corner, `flat` = 3 * triangle + corner: stored, or derived for a quad soup."""
    if mesh.quads != Int32(0):
        var t = flat // 3
        var k = flat - t * 3
        var base = (t >> 1) * 4
        if k == 0:
            return base
        return base + k + (t & 1)
    return Int(mesh.vertexIndices[unsafe_offset=flat])


@always_inline
def mesh_points_floats(n_verts: Int, quads: Int32) -> Int:
    """Floats in a mesh's position buffer: four a vertex, or for quads == 2 blocks of 256 vertices of 390 floats."""
    if quads == Int32(2):
        return ((n_verts + 255) >> 8) * 390
    return 4 * n_verts


@no_inline
def _quantised_pos(points: Pointer[Float32, MutUntrackedOrigin], v: Int) -> Vec3f:
    """A vertex of a quads == 2 buffer: blocks of 256 vertices, each a corner and a step (six floats) and 256
    triples of 16-bit steps from that corner; the parser quantises and refits the BVH to it."""
    var blk = v >> 8
    var h = points.unsafe_offset(blk * 390)
    var q = points.unsafe_bitcast[UInt16]()
    var o = blk * 780 + 12 + (v & 255) * 3
    return Vec3f(fma(Float32(q[unsafe_offset=o]), h[unsafe_offset=3], h[unsafe_offset=0]),
                 fma(Float32(q[unsafe_offset=o + 1]), h[unsafe_offset=4], h[unsafe_offset=1]),
                 fma(Float32(q[unsafe_offset=o + 2]), h[unsafe_offset=5], h[unsafe_offset=2]))


@always_inline
def mesh_pos(mesh: TriangleMesh, v: Int) -> Vec3f:
    """Position of vertex v (see _quantised_pos for the 16-bit form used by quad soups)."""
    if mesh.quads == Int32(2):
        return _quantised_pos(mesh.points, v)
    return Vec3f(mesh.points[unsafe_offset=v * 4], mesh.points[unsafe_offset=v * 4 + 1], mesh.points[unsafe_offset=v * 4 + 2])


@always_inline
def mesh_point(mesh: TriangleMesh, v: Int) -> Point3f:
    var p = mesh_pos(mesh, v)
    return Point3f(p.x, p.y, p.z)


@always_inline
def mesh_face(mesh: TriangleMesh, tri: Int) -> Int:
    """The triangle's face id (a Ptex face); a quad soup stores one a quad."""
    return Int(mesh.faceIndices[unsafe_offset=(tri >> 1) if mesh.quads != Int32(0) else tri])


@always_inline
def mesh_material(mesh: TriangleMesh, tri: Int) -> Int:
    """The triangle's material in a merged template; a quad soup stores one a quad."""
    return Int(mesh.materials[unsafe_offset=(tri >> 1) if mesh.quads != Int32(0) else tri])


def encode_normal(x: Float32, y: Float32, z: Float32) -> UInt32:
    """A direction as two signed 16-bit octahedral coordinates (about 0.002 degrees of error): the octahedron
    |x| + |y| + |z| = 1 unfolded to a square, the lower half folded outwards."""
    var s = abs(x) + abs(y) + abs(z)
    if s <= Float32(0):
        return UInt32(0)
    var px = x / s; var py = y / s
    if z < Float32(0):
        var qx = (Float32(1) - abs(py)) * (Float32(1) if px >= Float32(0) else Float32(-1))
        var qy = (Float32(1) - abs(px)) * (Float32(1) if py >= Float32(0) else Float32(-1))
        px = qx; py = qy
    var ix = Int(floor(px * Float32(32767) + Float32(0.5))); var iy = Int(floor(py * Float32(32767) + Float32(0.5)))
    return UInt32(ix & 0xFFFF) | (UInt32(iy & 0xFFFF) << 16)


@always_inline
def decode_normal(w: UInt32) -> Vec3f:
    var ix = Int(w & UInt32(0xFFFF)); var iy = Int(w >> 16)
    if ix >= 32768: ix -= 65536
    if iy >= 32768: iy -= 65536
    var x = Float32(ix) / Float32(32767); var y = Float32(iy) / Float32(32767)
    var z = Float32(1) - abs(x) - abs(y)
    var t = max(-z, Float32(0))
    x += -t if x >= Float32(0) else t
    y += -t if y >= Float32(0) else t
    var inv = Float32(1) / sqrt(x * x + y * y + z * z)
    return Vec3f(x * inv, y * inv, z * inv)


@always_inline
def mesh_normal(mesh: TriangleMesh, v: Int) -> Vec3f:
    """Shading normal of vertex v (the mesh must have normals)."""
    return decode_normal(mesh.normals.unsafe_bitcast[UInt32]()[unsafe_offset=v])


@always_inline
def mesh_uv(mesh: TriangleMesh, v: Int) -> Tuple[Float32, Float32]:
    """Texture coordinate of vertex v: stored, or the corner its index names (TriangleMesh.uv_corners)."""
    if mesh.uv_corners != Int32(0):
        var c = v & 3
        return (Float32(1) if (c == 1 or c == 2) else Float32(0), Float32(1) if c >= 2 else Float32(0))
    return (mesh.uvs[unsafe_offset=v * 2], mesh.uvs[unsafe_offset=v * 2 + 1])


@always_inline
def mesh_has_uvs(mesh: TriangleMesh) -> Bool:
    return mesh.uv_corners != Int32(0) or Int(mesh.uvs) > 4


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
    if mesh.uv_corners != Int32(0) or _is_real_ptr(mesh.uvs):
        su = w0*mesh_uv(mesh, v0)[0] + bu*mesh_uv(mesh, v1)[0] + bv*mesh_uv(mesh, v2)[0]
        tv = w0*mesh_uv(mesh, v0)[1] + bu*mesh_uv(mesh, v1)[1] + bv*mesh_uv(mesh, v2)[1]
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



