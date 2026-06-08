from std.ffi import external_call
from std.memory import alloc
from std.math import sqrt, acos, atan2, cos

# ── Math constants ─────────────────────────────────────────────────────────────
# Stored as Float32 aliases so every formula reads like the paper it came from.
# See: docs/01_geometry.md

# <<listing: Math Constants>>
alias PI         : Float32 = 3.14159265358979323846
alias TWO_PI     : Float32 = 6.28318530717958647692
alias INV_PI     : Float32 = 0.31830988618379067154
alias INV_TWO_PI : Float32 = 0.15915494309189533577
alias INV_FOUR_PI: Float32 = 0.07957747154594766788
alias SQRT2      : Float32 = 1.41421356237309504880
# <</listing>>

# ── Point3f / Vec3f ────────────────────────────────────────────────────────────
# Semantically distinct (affine point vs. free vector) but identical layout:
# 3 × Float32 = 12 bytes, packed, TrivialRegisterPassable.
# Using named structs instead of SIMD[f32,3] avoids the hidden power-of-2
# lane padding that would bloat BVH2Node from 32 to 40 bytes.
# Matches pbrt-v4 naming convention.
# See: docs/01_geometry.md

@fieldwise_init
# <<listing: Point3f>>
struct Point3f(TrivialRegisterPassable):
    """An affine point in 3D space (position, not a direction)."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    fn __add__(self, v: Vec3f) -> Point3f:
        return Point3f(self.x + v.x, self.y + v.y, self.z + v.z)

    @always_inline
    fn __sub__(self, o: Point3f) -> Vec3f:
        """Point minus point yields a displacement vector."""
        return Vec3f(self.x - o.x, self.y - o.y, self.z - o.z)

    @always_inline
    fn to_simd(self) -> SIMD[DType.float32, 3]:
        return SIMD[DType.float32, 3](self.x, self.y, self.z)

@fieldwise_init
# <</listing>>
struct Vec3f(TrivialRegisterPassable):
# <<listing: Vec3f>>
    """A free vector in 3D space (direction, displacement, or surface normal)."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    fn __neg__(self) -> Vec3f:
        return Vec3f(-self.x, -self.y, -self.z)

    @always_inline
    fn __add__(self, b: Vec3f) -> Vec3f:
        return Vec3f(self.x + b.x, self.y + b.y, self.z + b.z)

    @always_inline
    fn __sub__(self, b: Vec3f) -> Vec3f:
        return Vec3f(self.x - b.x, self.y - b.y, self.z - b.z)

    @always_inline
    fn __mul__(self, s: Float32) -> Vec3f:
        return Vec3f(self.x * s, self.y * s, self.z * s)

    @always_inline
    fn __truediv__(self, s: Float32) -> Vec3f:
        var inv = Float32(1.0) / s
        return Vec3f(self.x * inv, self.y * inv, self.z * inv)

    @always_inline
    fn length_sq(self) -> Float32:
        """Squared length — avoids a sqrt when only ordering matters."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    @always_inline
    fn length(self) -> Float32:
        return sqrt(self.length_sq())

    @always_inline
    fn normalize(self) -> Vec3f:
        """Returns a unit vector in the same direction."""
        return self * (Float32(1.0) / self.length())

    @always_inline
    fn dot(self, b: Vec3f) -> Float32:
        return self.x * b.x + self.y * b.y + self.z * b.z

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

# <</listing>>
# ── Color / Spectrum ───────────────────────────────────────────────────────────
# SampledSpectrum is currently an RGB triple. This alias makes the shader code
# forward-compatible with a future spectral representation.
# See: docs/02_spectra_and_color.md

@fieldwise_init
struct RGB(TrivialRegisterPassable):
    """Linear-light RGB colour value. All arithmetic is in scene-linear space."""
# <<listing: RGB>>
    var r: Float32
    var g: Float32
    var b: Float32

    @always_inline
    fn __add__(self, o: RGB) -> RGB:
        return RGB(self.r + o.r, self.g + o.g, self.b + o.b)

    @always_inline
    fn __mul__(self, o: RGB) -> RGB:
        return RGB(self.r * o.r, self.g * o.g, self.b * o.b)

    @always_inline
    fn __mul__(self, s: Float32) -> RGB:
        return RGB(self.r * s, self.g * s, self.b * s)

    @always_inline
    fn __truediv__(self, s: Float32) -> RGB:
        var inv = Float32(1.0) / s
        return RGB(self.r * inv, self.g * inv, self.b * inv)

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
    fn is_black(self) -> Bool:
        """Returns True when all channels are zero or negative."""
        return self.r <= Float32(0.0) and self.g <= Float32(0.0) and self.b <= Float32(0.0)

    @always_inline
    fn luma(self) -> Float32:
        """CIE Y luminance (Rec. 709 primaries)."""
        return Float32(0.2126) * self.r + Float32(0.7152) * self.g + Float32(0.0722) * self.b

    @always_inline
    fn clamp(self, lo: Float32, hi: Float32) -> RGB:
        """Per-channel clamp — useful for tonemapping guard values."""
# <</listing>>
        var clamp_f = fn(v: Float32) -> Float32: return lo if v < lo else (hi if v > hi else v)
        return RGB(clamp_f(self.r), clamp_f(self.g), clamp_f(self.b))

alias SampledSpectrum = RGB
# <<listing: SampledSpectrum>>
"""Placeholder alias for a future spectral type. Currently == RGB.
   See: docs/02_spectra_and_color.md
# <</listing>>
"""

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
    var medium_interface_idx: Int32  # -1 = no medium interface bound

@fieldwise_init
struct TriangleMesh_C(TrivialRegisterPassable):
    var points: UnsafePointer[Float32, MutAnyOrigin]
    var faceIndices: UnsafePointer[Int64, MutAnyOrigin]
    var vertexIndices: UnsafePointer[Int64, MutAnyOrigin]
    var uvs: UnsafePointer[Float32, MutAnyOrigin]   # nullable; stride 2 floats per vertex

# ── Ray ───────────────────────────────────────────────────────────────────────
# See: docs/03_shapes_and_acceleration.md

@fieldwise_init
struct Ray_C(TrivialRegisterPassable):
    """A ray: a world-space origin point and a unit direction vector."""
    var origin: Point3f
# <<listing: Ray_C>>
    var direction: Vec3f

# ── Intersection ──────────────────────────────────────────────────────────────
# <</listing>>

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
# See: docs/07_path_tracing.md

@fieldwise_init
struct PathState_C(TrivialRegisterPassable):
# <<listing: PathState_C>>
    var ray: Ray_C
    var throughput: SampledSpectrum
    var estimate: SampledSpectrum
    var albedo: SampledSpectrum
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
    var current_medium_idx: Int32  # -1 = vacuum; >= 0 = index into scene.mediums
# <</listing>>

# ── Lights ────────────────────────────────────────────────────────────────────
# See: docs/06_lights_and_materials.md

@fieldwise_init
struct AreaLight_C(TrivialRegisterPassable):
    var meshIdx: Int32
    var n_tris: Int32       # number of triangles in this light mesh
    var emission: SampledSpectrum
    var total_area: Float32 # total surface area of this light mesh

@fieldwise_init
struct Sphere_C(TrivialRegisterPassable):
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
    var emission: SampledSpectrum


# ── Homogeneous media ──────────────────────────────────────────────────────────

@fieldwise_init
struct Medium_C(TrivialRegisterPassable):
    """Homogeneous participating medium (PBRT-v4 HomogeneousMedium).
    sigma_a + sigma_s are pre-scaled by 'scale'.
    g = Henyey-Greenstein anisotropy in [-1, 1]; 0 = isotropic.
    """
    var sigma_a: SampledSpectrum   # absorption coefficient (1/m)
    var sigma_s: SampledSpectrum   # scattering coefficient (1/m)
    var g:       Float32           # HG anisotropy
    var _pad0:   Float32
    var _pad1:   Float32
    var _pad2:   Float32

@fieldwise_init
struct MediumInterface_C(TrivialRegisterPassable):
    """Binds inside/outside media to a surface. -1 = vacuum."""
    var inside_medium_idx:  Int32
    var outside_medium_idx: Int32


@fieldwise_init
struct DistantLight_C(TrivialRegisterPassable):
    """A directional (infinite-distance) light.
    `direction` points FROM the light TOWARD the scene (world space).
    """
    var direction: Vec3f
    var _pad: Float32
    var emission: SampledSpectrum
    var _pad2: Float32

@fieldwise_init
struct PointLight_C(TrivialRegisterPassable):
    """An isotropic point light at a world-space position."""
    var position: Point3f
    var _pad: Float32
    var intensity: SampledSpectrum
    var _pad2: Float32

@fieldwise_init
struct InfiniteLight_C(TrivialRegisterPassable):
    """An environment map (lat-long HDRI), importance-sampled via 2D CDF.
    See: docs/06_lights_and_materials.md — Infinite Area Lights.
    """
    var scale: SampledSpectrum
    var tex_idx: Int32   # -1 = solid colour, >= 0 = texture
    var cdf_w: Int32     # env-map pixel width (also CDF width; 0 = no texture)
    var cdf_h: Int32     # env-map pixel height
    var cdf_ptr: UnsafePointer[Float32, MutAnyOrigin]   # flat 2D CDF (marginal + conditional)
    var pixels_ptr: UnsafePointer[Float32, MutAnyOrigin] # raw HDR pixels, 3 floats/pixel (CPU only)
    var world_to_light: UnsafePointer[Float32, MutAnyOrigin]  # 16-float col-major inverse of light CTM

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
    var contrib: SampledSpectrum
    var active: Int32
    var _pad: Int32


@fieldwise_init
struct TileResult_C(TrivialRegisterPassable):
    var estimate: SampledSpectrum
    var albedo: SampledSpectrum
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32

# ── Frame (local shading coordinate system) ───────────────────────────────────
# An orthonormal basis where z aligns with the surface shading normal.
# Used throughout the BSDF implementations to convert between world space
# and the hemisphere convention where cos θ = v.z.
#
# tangent-frame construction uses Duff et al. 2017,
# "Building an Orthonormal Basis, Revisited", JCGT 6(1).
# The key insight: sign = sign(n.z) makes the formula branchless and
# numerically stable even at the south pole (n.z ≈ -1).
# See: docs/05_reflection_models.md — Local Frames.

@fieldwise_init
struct Frame(TrivialRegisterPassable):
# <<listing: Frame>>
    """Orthonormal shading frame. z = shading normal (hemisphere up).
    x = tangent, y = bitangent.
    See: docs/05_reflection_models.md
    """
    var x: Vec3f   # tangent
    var y: Vec3f   # bitangent
    var z: Vec3f   # normal (hemisphere axis)

    @staticmethod
    @always_inline
    fn from_z(n: Vec3f) -> Frame:
        """Build a frame from a single normal vector (Duff et al. 2017).
        Branchless, GPU-friendly, stable for all input directions.
        """
        var sign = Float32(1.0) if n.z >= Float32(0.0) else Float32(-1.0)
        var a = Float32(-1.0) / (sign + n.z)
        var b = n.x * n.y * a
        var tangent   = Vec3f(Float32(1.0) + sign * n.x * n.x * a, sign * b, -sign * n.x)
        var bitangent = Vec3f(b, sign + n.y * n.y * a, -n.y)
        return Frame(tangent, bitangent, n)

    @always_inline
    fn to_local(self, v: Vec3f) -> Vec3f:
        """Project world-space vector into the local frame (x,y,z coordinates)."""
        return Vec3f(v.dot(self.x), v.dot(self.y), v.dot(self.z))

    @always_inline
    fn to_world(self, v: Vec3f) -> Vec3f:
        """Reconstruct a world-space vector from local-frame coordinates."""
        return self.x * v.x + self.y * v.y + self.z * v.z

# ── Geometry helpers ───────────────────────────────────────────────────────────
# See: docs/01_geometry.md
# <</listing>>

@always_inline
fn safe_sqrt(x: Float32) -> Float32:
# <<listing: safe_sqrt>>
    """sqrt(max(x, 0)) — avoids NaN from small negative values due to rounding."""
    return sqrt(x if x > Float32(0.0) else Float32(0.0))

# <</listing>>
@always_inline
# <<listing: reflect>>
fn reflect(wo: Vec3f, n: Vec3f) -> Vec3f:
    """Specular reflection of wo about surface normal n.
    Equation: wi = 2(wo·n)n − wo.
    Both wo and n should point away from the surface.
    See: docs/05_reflection_models.md — Specular Reflection.
    """
    return n * (Float32(2.0) * wo.dot(n)) - wo

@always_inline
# <</listing>>
fn refract(wi: Vec3f, n: Vec3f, eta: Float32) -> Tuple[Bool, Vec3f]:
# <<listing: refract>>
    """Snell's law refraction. eta = η_i / η_t.
    Returns (True, wt) on success, (False, _) on total internal reflection.
    See: docs/05_reflection_models.md — Specular Transmission.
    """
    var cos_theta_i = n.dot(wi)
    var sin2_theta_i = max(Float32(0.0), Float32(1.0) - cos_theta_i * cos_theta_i)
    var sin2_theta_t = eta * eta * sin2_theta_i
    if sin2_theta_t >= Float32(1.0):
        return (False, Vec3f(0.0, 0.0, 0.0))   # total internal reflection
    var cos_theta_t = safe_sqrt(Float32(1.0) - sin2_theta_t)
    var wt = wi * (-eta) + n * (eta * cos_theta_i - cos_theta_t)
# <</listing>>
    return (True, wt)
# <<listing: schlick_fresnel>>

@always_inline
fn schlick_fresnel(cos_theta: Float32, f0: Float32) -> Float32:
    """Schlick (1994) approximation to the Fresnel reflectance.
    f0 = ((η−1)/(η+1))² is the normal-incidence reflectance.
    See: docs/05_reflection_models.md — Fresnel.
    """
    var t = Float32(1.0) - cos_theta
    var t2 = t * t
    return f0 + (Float32(1.0) - f0) * (t2 * t2 * t)
# <</listing>>

@always_inline
fn spherical_direction(sin_theta: Float32, cos_theta: Float32, phi: Float32) -> Vec3f:
    """Convert spherical coordinates (θ,φ) to a unit Cartesian vector.
    Convention: y = up (cos θ), xz = equatorial plane.
    See: docs/01_geometry.md — Spherical Coordinates.
    """
    var sin_phi: Float32
    var cos_phi: Float32
    # Manual sin/cos via identity avoids an extra import
    sin_phi = sqrt(max(Float32(0.0), Float32(1.0) - cos(phi)*cos(phi)))
    cos_phi = cos(phi)
    return Vec3f(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi)

@always_inline
fn spherical_theta(v: Vec3f) -> Float32:
    """Polar angle θ ∈ [0, π] of a unit vector (y = up convention)."""
    return acos(max(Float32(-1.0), min(Float32(1.0), v.y)))

@always_inline
fn spherical_phi(v: Vec3f) -> Float32:
    """Azimuthal angle φ ∈ [0, 2π] of a unit vector (y = up convention)."""
    var p = atan2(v.z, v.x)
    return p if p >= Float32(0.0) else p + TWO_PI

# ── SIMD math helpers (used in BVH and shading hot paths) ────────────────────

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
