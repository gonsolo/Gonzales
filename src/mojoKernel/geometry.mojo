from std.ffi import external_call
from std.memory import alloc

# ── Primitive math types ───────────────────────────────────────────────────────
# Point3f and Vec3f are semantically distinct (affine point vs. free vector)
# but share an identical memory layout: 3 × Float32 = 12 bytes, packed.
# This matches pbrt-v4 convention. Using named structs instead of SIMD[f32,3]
# avoids the hidden 4th lane padding that SIMD[f32,N] requires (N must be a
# power of 2), which would break cache-line-sized BVH node packing.

@fieldwise_init
struct Point3f(TrivialRegisterPassable):
    """An affine point in 3D space (position)."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    fn __add__(self, v: Vec3f) -> Point3f:
        return Point3f(self.x + v.x, self.y + v.y, self.z + v.z)

    @always_inline
    fn __sub__(self, o: Point3f) -> Vec3f:
        return Vec3f(self.x - o.x, self.y - o.y, self.z - o.z)

    @always_inline
    fn to_simd(self) -> SIMD[DType.float32, 3]:
        return SIMD[DType.float32, 3](self.x, self.y, self.z)

@fieldwise_init
struct Vec3f(TrivialRegisterPassable):
    """A free vector in 3D space (direction / displacement)."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    fn __neg__(self) -> Vec3f:
        return Vec3f(-self.x, -self.y, -self.z)

    @always_inline
    fn __mul__(self, s: Float32) -> Vec3f:
        return Vec3f(self.x * s, self.y * s, self.z * s)

    @always_inline
    fn to_simd(self) -> SIMD[DType.float32, 3]:
        return SIMD[DType.float32, 3](self.x, self.y, self.z)

@always_inline
fn vec3f(s: SIMD[DType.float32, 3]) -> Vec3f:
    """Convert a SIMD[f32,3] to a Vec3f."""
    return Vec3f(s[0], s[1], s[2])

@always_inline
fn point3f(s: SIMD[DType.float32, 3]) -> Point3f:
    """Convert a SIMD[f32,3] to a Point3f."""
    return Point3f(s[0], s[1], s[2])

# ── Color ──────────────────────────────────────────────────────────────────────

@fieldwise_init
struct RGB(TrivialRegisterPassable):
    var r: Float32
    var g: Float32
    var b: Float32

    @always_inline
    fn __mul__(self, o: RGB) -> RGB:
        return RGB(self.r * o.r, self.g * o.g, self.b * o.b)

    @always_inline
    fn __mul__(self, s: Float32) -> RGB:
        return RGB(self.r * s, self.g * s, self.b * s)

    @always_inline
    fn __imul__(mut self, o: RGB):
        self.r *= o.r; self.g *= o.g; self.b *= o.b

    @always_inline
    fn __imul__(mut self, s: Float32):
        self.r *= s; self.g *= s; self.b *= s

    @always_inline
    fn __iadd__(mut self, o: RGB):
        self.r += o.r; self.g += o.g; self.b += o.b

    @always_inline
    fn luma(self) -> Float32:
        return Float32(0.2126)*self.r + Float32(0.7152)*self.g + Float32(0.0722)*self.b

# ── Scene primitives ───────────────────────────────────────────────────────────

@fieldwise_init
struct PrimId_C(TrivialRegisterPassable):
    var id1: Int64
    var id2: Int64
    var materialIndex: Int64
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var _pad3: Int8
    var _pad4: Int8
    var _pad5: Int8
    var _pad6: Int8

@fieldwise_init
struct Material_C(TrivialRegisterPassable):
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var albedo: RGB
    var emission: RGB
    var tex_idx: Int32      # -1 = no texture; >= 0 = index into texture table
    var roughU: Float32     # GGX uroughness (conductor); 0 = perfect mirror
    var roughV: Float32     # GGX vroughness (conductor); 0 = perfect mirror
    var normal_tex_idx: Int32  # -1 = no normal map; >= 0 = index into texture table

@fieldwise_init
struct TriangleMesh_C(TrivialRegisterPassable):
    var points: UnsafePointer[Float32, MutAnyOrigin]
    var faceIndices: UnsafePointer[Int64, MutAnyOrigin]
    var vertexIndices: UnsafePointer[Int64, MutAnyOrigin]
    var uvs: UnsafePointer[Float32, MutAnyOrigin]   # nullable; stride 2 floats per vertex

# ── Ray ───────────────────────────────────────────────────────────────────────

@fieldwise_init
struct Ray_C(TrivialRegisterPassable):
    """A ray: a 3D origin point and a unit direction vector."""
    var origin: Point3f
    var direction: Vec3f

# ── Intersection ──────────────────────────────────────────────────────────────

@fieldwise_init
struct Intersection_C(TrivialRegisterPassable):
    var primId: PrimId_C
    var tHit: Float32
    var u: Float32
    var v: Float32
    var hit: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8

# ── Path state ────────────────────────────────────────────────────────────────

@fieldwise_init
struct PathState_C(TrivialRegisterPassable):
    var ray: Ray_C
    var throughput: RGB
    var estimate: RGB
    var albedo: RGB
    var bounce: Int32
    var pcgState: UInt64
    var pcgInc: UInt64
    var active: Int8
    var specularBounce: Int8   # 1 if previous scatter was a delta BSDF (mirror/glass)
    var _pad2: Int8
    var _pad3: Int8
    # lastBsdfPdf: cosine-hemisphere PDF from the previous scatter (cos_theta / pi).
    # Used for MIS weighting when the next bounce hits an emitter.
    var lastBsdfPdf: Float32

# ── Lights ────────────────────────────────────────────────────────────────────

@fieldwise_init
struct AreaLight_C(TrivialRegisterPassable):
    var meshIdx: Int32
    var n_tris: Int32       # number of triangles in this light mesh
    var emission: RGB
    var total_area: Float32 # total surface area of this light mesh

@fieldwise_init
struct DistantLight_C(TrivialRegisterPassable):
    """A directional (infinite) light. direction points FROM the light toward the scene."""
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
    var scale: RGB
    var tex_idx: Int32   # -1 = solid colour, >= 0 = texture
    var cdf_w: Int32     # env-map CDF width (0 = no CDF)
    var cdf_h: Int32     # env-map CDF height
    var cdf_ptr: UnsafePointer[Float32, MutAnyOrigin]  # flat 2D CDF (marginal + conditional)

# ── GPU / render pipeline helpers ─────────────────────────────────────────────

@fieldwise_init
struct GpuTexture_C(TrivialRegisterPassable):
    var data: UnsafePointer[Float32, MutAnyOrigin]  # device pointer, pre-linearised float RGB
    var width: Int32
    var height: Int32

@fieldwise_init
struct ShadowTask_C(TrivialRegisterPassable):
    """A deferred shadow ray with its pre-computed radiance contribution."""
    var origin: Point3f
    var direction: Vec3f
    var tmax: Float32
    var contrib: RGB
    var active: Int32
    var _pad: Int32

@fieldwise_init
struct PixelSample_C(TrivialRegisterPassable):
    var filmX: Float32
    var filmY: Float32
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32
    var _pad: Int32
    var pcgState: UInt64
    var pcgInc: UInt64

@fieldwise_init
struct TileResult_C(TrivialRegisterPassable):
    var estimate: RGB
    var albedo: RGB
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32

# ── Math helpers ──────────────────────────────────────────────────────────────

@always_inline
fn cross(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 3]:
    var a_yzx = SIMD[DType.float32, 3](a[1], a[2], a[0])
    var b_zxy = SIMD[DType.float32, 3](b[2], b[0], b[1])
    var a_zxy = SIMD[DType.float32, 3](a[2], a[0], a[1])
    var b_yzx = SIMD[DType.float32, 3](b[1], b[2], b[0])
    return a_yzx * b_zxy - a_zxy * b_yzx

@always_inline
fn dot(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> Float32:
    var prod = a * b
    return prod[0] + prod[1] + prod[2]

@always_inline
fn intersect_triangle(
    ray_org: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
    p0: SIMD[DType.float32, 3],
    p1: SIMD[DType.float32, 3],
    p2: SIMD[DType.float32, 3],
    tMax: Float32
) -> Tuple[Bool, Float32, Float32, Float32]:
    var e1 = p1 - p0
    var e2 = p2 - p0
    var pvec = cross(ray_dir, e2)
    var det = dot(e1, pvec)

    if det > -0.0000001 and det < 0.0000001:
        return (False, tMax, 0.0, 0.0)

    var invDet = 1.0 / det
    var tvec = ray_org - p0
    var u = dot(tvec, pvec) * invDet

    if u < 0.0 or u > 1.0:
        return (False, tMax, 0.0, 0.0)

    var qvec = cross(tvec, e1)
    var v = dot(ray_dir, qvec) * invDet

    if v < 0.0 or u + v > 1.0:
        return (False, tMax, 0.0, 0.0)

    var t = dot(e2, qvec) * invDet
    if t <= 0.0 or t > tMax:
        return (False, tMax, 0.0, 0.0)

    return (True, t, u, v)
