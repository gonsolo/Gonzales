from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.memory import bitcast
from std.math import sqrt, acos, atan2, cos, sin, min, max, abs, floor, log, exp
from std.sys.info import align_of
from gonzales.spectrum import SampledWavelengths, SpectralSample, spec_refl, spec_refl_unbounded, rgb_illuminant_to_spectral_sample
from gonzales.nanovdb import nvdb_sample_index, nvdb_majorant_at, nvdb_leaf_base, nvdb_leaf_value
from gonzales.rng import PCG32

# Value structs shared with GPU code can't hold Optional[Pointer], so an
# "unset" pointer field is instead left at its `.unsafe_dangling()` sentinel --
# which Mojo returns as an address equal to align_of[T](), NOT a fixed value
# across pointee types (4 for Float32, 1 for UInt8, 8 for a pointer-sized
# struct, ...). Use this helper instead of a hand-picked magic-number
# threshold at each call site.
@always_inline
def _is_real_ptr[T: AnyType, O: Origin[mut=True]](ptr: Pointer[T, O]) -> Bool:
    return Int(ptr) > align_of[T]()

# ── Math constants ─────────────────────────────────────────────────────────────
# Stored as Float32 aliases so every formula reads like the paper it came from.
# See: docs/01_geometry.md

# <<listing: Math Constants>>
comptime PI         : Float32 = 3.14159265358979323846
comptime TWO_PI     : Float32 = 6.28318530717958647692
comptime INV_PI     : Float32 = 0.31830988618379067154
comptime INV_FOUR_PI: Float32 = 0.07957747154594766788


# A maxdepth cap does not kill a path outright: pbrt checks `depth++ >=
# maxDepth` AFTER collecting this vertex's own emission and BEFORE sampling
# a new direction (integrators.cpp), so a path already at its full bounce
# budget still gets ONE MORE segment traced -- the ray already fired from
# its last real scatter -- which may simply escape to an infinite light or
# land on an emitter, before refusing to scatter again. That grace period
# needs a round/iteration to run in; it is not automatic. Three call sites
# each enforce it in their own loop's native shape (there is no shared
# control flow to factor it into -- rendering.mojo's CPU host loop marks a
# per-path `at_cap` flag consumed by _shade_dispatch, gpu.mojo's GPU round
# budget just needs +1 dispatched round, sppm.mojo's monolithic bounce+
# dispatch loop needs an inline `bounce > max_charged` guard before its
# material dispatch), but they all owe their existence to this one constant
# and this one reason -- see project_gpu_pt_terminal_round_bug memory for
# the repro (barcelona's double-glazed window, needing exactly 5 real
# bounces to reach the sky) that found the GPU PT and SPPM instances of a
# bug rendering.mojo's CPU path had already fixed once (583a3390).
comptime TERMINAL_SEGMENT_GRACE_ROUNDS = 1

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
struct Point3f(TrivialRegisterPassable, Writable):
    """An affine point in 3D space (position, not a direction)."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    def __init__(out self, v: Float32):
        """Broadcast constructor — Point3f(0) for the origin."""
        self.x = v; self.y = v; self.z = v

    @always_inline
    def __add__(self, v: Vec3f) -> Point3f:
        return Point3f(self.x + v.x, self.y + v.y, self.z + v.z)

    @always_inline
    def __sub__(self, o: Point3f) -> Vec3f:
        """Point minus point yields a displacement vector."""
        return Vec3f(self.x - o.x, self.y - o.y, self.z - o.z)

    @always_inline
    def to_simd(self) -> Vec3f:
        return Vec3f(self.x, self.y, self.z)

    @always_inline
    def __getitem__(self, i: Int) -> Float32:
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    @always_inline
    def __setitem__(mut self, i: Int, v: Float32):
        if i == 0:
            self.x = v
        elif i == 1:
            self.y = v
        else:
            self.z = v

@fieldwise_init
# <</listing>>
struct Vec3f(TrivialRegisterPassable, Writable):
# <<listing: Vec3f>>
    """A free vector in 3D space (direction, displacement, or surface normal)."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    def __init__(out self, v: Float32):
        """Broadcast constructor — Vec3f(0) for the zero vector."""
        self.x = v; self.y = v; self.z = v

    @always_inline
    def __neg__(self) -> Vec3f:
        return Vec3f(-self.x, -self.y, -self.z)

    @always_inline
    def __add__(self, b: Vec3f) -> Vec3f:
        return Vec3f(self.x + b.x, self.y + b.y, self.z + b.z)

    @always_inline
    def __sub__(self, b: Vec3f) -> Vec3f:
        return Vec3f(self.x - b.x, self.y - b.y, self.z - b.z)

    @always_inline
    def __mul__(self, s: Float32) -> Vec3f:
        return Vec3f(self.x * s, self.y * s, self.z * s)

    @always_inline
    def __rmul__(self, s: Float32) -> Vec3f:
        return Vec3f(self.x * s, self.y * s, self.z * s)

    @always_inline
    def __truediv__(self, s: Float32) -> Vec3f:
        var inv = Float32(1.0) / s
        return Vec3f(self.x * inv, self.y * inv, self.z * inv)

    @always_inline
    def __iadd__(mut self, b: Vec3f):
        self.x += b.x; self.y += b.y; self.z += b.z

    @always_inline
    def __isub__(mut self, b: Vec3f):
        self.x -= b.x; self.y -= b.y; self.z -= b.z

    @always_inline
    def __imul__(mut self, s: Float32):
        self.x *= s; self.y *= s; self.z *= s

    @always_inline
    def __itruediv__(mut self, s: Float32):
        var inv = Float32(1.0) / s
        self.x *= inv; self.y *= inv; self.z *= inv

    @always_inline
    def length_sq(self) -> Float32:
        """Squared length — avoids a sqrt when only ordering matters."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    @always_inline
    def length(self) -> Float32:
        return sqrt(self.length_sq())

    @always_inline
    def normalize(self) -> Vec3f:
        """Returns a unit vector in the same direction."""
        return self * (Float32(1.0) / self.length())

    @always_inline
    def dot(self, b: Vec3f) -> Float32:
        return self.x * b.x + self.y * b.y + self.z * b.z

    @always_inline
    def to_simd(self) -> Vec3f:
        return Vec3f(self.x, self.y, self.z)

    @always_inline
    def __getitem__(self, i: Int) -> Float32:
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    @always_inline
    def __setitem__(mut self, i: Int, v: Float32):
        if i == 0:
            self.x = v
        elif i == 1:
            self.y = v
        else:
            self.z = v

    @always_inline
    def __mul__(self, b: Vec3f) -> Vec3f:
        """Elementwise (Hadamard) product -- matches SIMD[f32,3]'s own `*`."""
        return Vec3f(self.x * b.x, self.y * b.y, self.z * b.z)

@always_inline
def vec3f(s: Vec3f) -> Vec3f:
    """Convert a SIMD[f32,3] to a Vec3f."""
    return Vec3f(s[0], s[1], s[2])

@always_inline
def point3f(s: Vec3f) -> Point3f:
    """Convert a SIMD[f32,3] to a Point3f."""
    return Point3f(s[0], s[1], s[2])

@always_inline
def store_vec3[O: Origin[mut=True]](dst: Pointer[Float32, O], slot: Int, v: Vec3f):
    """Write a Point3f/Vec3f (pass `.to_simd()`) into a flat, stride-3
    Float32 buffer at `slot` -- i.e. dst[slot*3 : slot*3+3] -- replacing the
    dst[slot*3+0]=v.x; dst[slot*3+1]=v.y; dst[slot*3+2]=v.z pattern repeated
    at per-pixel G-buffer-style write sites."""
    dst[unsafe_offset=slot*3+0] = v[0]
    dst[unsafe_offset=slot*3+1] = v[1]
    dst[unsafe_offset=slot*3+2] = v[2]

# <</listing>>

@fieldwise_init
struct Point2f(TrivialRegisterPassable):
    """A 2D point: texture UV coordinates, or a [0,1]^2 sample-space pair
    (e.g. two independent random numbers used together for a disk/hemisphere/
    CDF sample). Matches pbrt-v4 naming convention."""
    var x: Float32
    var y: Float32

    @always_inline
    def __init__(out self, v: Float32):
        """Broadcast constructor — Point2f(0) for the origin."""
        self.x = v; self.y = v

@fieldwise_init
struct Point2i(TrivialRegisterPassable):
    """A 2D INTEGER point: a pixel coordinate carried as one value instead of
    two loose Int32s. Narrow in scope today (restir_jitter_pixel below is its
    only producer) -- the flat `px = tid % fw; py = tid // fw` GPU
    thread-index unpacks scattered through gpu.mojo/bdpt.mojo/sppm.mojo stay
    scalar deliberately (they feed direct flat-array indexing in a hot
    kernel), so this is not meant to replace those."""
    var x: Int32
    var y: Int32

@always_inline
def restir_jitter_pixel(center: Point2i, ang: Float32, rad: Float32) -> Point2i:
    """One candidate neighbour pixel for spatial reuse: `center` plus a
    disk-sampled offset at angle `ang`, radius `rad` (both already drawn by
    the caller — e.g. `ang = u*2pi`, `rad = sqrt(u2)*radius_px`). Byte-for-
    byte the same two lines that were independently copy-pasted into
    restir_gi.mojo, restir_vol.mojo, restir_sms.mojo and shading.mojo's own
    ReSTIR spatial-reuse loops -- the caller still does its own bounds check
    and flat-index conversion, since those differ per call site's own G-buffer
    layout."""
    return Point2i(center.x + Int32(cos(ang) * rad), center.y + Int32(sin(ang) * rad))

@fieldwise_init
struct Bounds3f(TrivialRegisterPassable):
    """An axis-aligned bounding box (world space)."""
    var min: Point3f
    var max: Point3f

# ── Color / Spectrum ───────────────────────────────────────────────────────────
# RGB is used where a quantity genuinely HAS three authored channels: a
# light's emission/intensity/scale, a medium's sigma_a/sigma_s, the denoiser's
# albedo AOV. Radiance transport itself is spectral -- spectrum.mojo's
# SpectralSample, four hero wavelengths -- in all three integrators (see
# project_spectral_throughput_flip).
#
# There used to be a `SampledSpectrum = RGB` alias here, described as the
# single switch a future spectral flip would throw. The flip happened without
# it, needed far more than a rename, and afterwards the alias only misled: a
# type named SampledSpectrum that was three floats, sitting on medium
# coefficients inside a renderer whose transport is four hero wavelengths. It
# is deleted. A type must not claim to be spectral while being RGB.
#
# NOTE for anyone converting one of these to spectral: an RGB *coefficient*
# (a medium's extinction, a single-scattering albedo) must be BAND-PICKED via
# rgb_bands_to_spectral_sample, never pushed through the reflectance
# upsampler -- RGB(a,a,a) does not come back as `a` in every lane.
# See: docs/02_spectra_and_color.md, "A coefficient is not a color"

@fieldwise_init
struct RGB(TrivialRegisterPassable):
    """Linear-light RGB colour value. All arithmetic is in scene-linear space."""
# <<listing: RGB>>
    var r: Float32
    var g: Float32
    var b: Float32

    @always_inline
    def __init__(out self, v: Float32):
        """Broadcast constructor — RGB(0) for black, RGB(1) for white."""
        self.r = v; self.g = v; self.b = v

    @always_inline
    def __add__(self, o: RGB) -> RGB:
        return RGB(self.r + o.r, self.g + o.g, self.b + o.b)

    @always_inline
    def __sub__(self, o: RGB) -> RGB:
        return RGB(self.r - o.r, self.g - o.g, self.b - o.b)

    @always_inline
    def __mul__(self, o: RGB) -> RGB:
        return RGB(self.r * o.r, self.g * o.g, self.b * o.b)

    @always_inline
    def __mul__(self, s: Float32) -> RGB:
        return RGB(self.r * s, self.g * s, self.b * s)

    @always_inline
    def __truediv__(self, s: Float32) -> RGB:
        var inv = Float32(1.0) / s
        return RGB(self.r * inv, self.g * inv, self.b * inv)

    @always_inline
    def __imul__(mut self, o: RGB):
        self.r *= o.r; self.g *= o.g; self.b *= o.b

    @always_inline
    def __imul__(mut self, s: Float32):
        self.r *= s; self.g *= s; self.b *= s

    @always_inline
    def __iadd__(mut self, o: RGB):
        self.r += o.r; self.g += o.g; self.b += o.b

    @always_inline
    def sum(self) -> Float32:
        """Sum of the three channels — e.g. for a squared-distance reduction."""
        return self.r + self.g + self.b

    @always_inline
    def is_black(self) -> Bool:
        """Returns True when all channels are zero or negative."""
        return self.r <= Float32(0.0) and self.g <= Float32(0.0) and self.b <= Float32(0.0)

    @always_inline
    def luma(self) -> Float32:
        """CIE Y luminance (Rec. 709 primaries)."""
        return Float32(0.2126) * self.r + Float32(0.7152) * self.g + Float32(0.0722) * self.b

    @always_inline
    def clamp(self, lo: Float32, hi: Float32) -> RGB:
        """Per-channel clamp — useful for tonemapping guard values."""
# <</listing>>
        return RGB(
            max(lo, min(hi, self.r)),
            max(lo, min(hi, self.g)),
            max(lo, min(hi, self.b)),
        )


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
    def from_z(n: Vec3f) -> Frame:
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
    def to_local(self, v: Vec3f) -> Vec3f:
        """Project world-space vector into the local frame (x,y,z coordinates)."""
        return Vec3f(v.dot(self.x), v.dot(self.y), v.dot(self.z))

    @always_inline
    def to_world(self, v: Vec3f) -> Vec3f:
        """Reconstruct a world-space vector from local-frame coordinates."""
        return self.x * v.x + self.y * v.y + self.z * v.z

# ── Geometry helpers ───────────────────────────────────────────────────────────
# See: docs/01_geometry.md
# <</listing>>

@always_inline
def safe_sqrt(x: Float32) -> Float32:
# <<listing: safe_sqrt>>
    """sqrt(max(x, 0)) — avoids NaN from small negative values due to rounding."""
    return sqrt(x if x > Float32(0.0) else Float32(0.0))

# <</listing>>
@always_inline
# <<listing: reflect>>
def reflect(wo: Vec3f, n: Vec3f) -> Vec3f:
    """Specular reflection of wo about surface normal n.
    Equation: wi = 2(wo·n)n − wo.
    Both wo and n should point away from the surface.
    See: docs/05_reflection_models.md — Specular Reflection.
    """
    return n * (Float32(2.0) * wo.dot(n)) - wo

@always_inline
# <</listing>>
def refract(wi: Vec3f, n: Vec3f, eta: Float32) -> Tuple[Bool, Vec3f]:
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
@always_inline
def spherical_direction(sin_theta: Float32, cos_theta: Float32, phi: Float32) -> Vec3f:
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
def _atan2f(y: Float32, x: Float32) -> Float32:
    """atan2 via minimax polynomial — avoids an unresolved CUDA libdevice
    extern this toolchain can't resolve for std.math.atan2 on GPU (see
    project_gpu_ptx_environment_break memory)."""
    var ax = abs(x); var ay = abs(y)
    var mn = min(ax, ay)
    var mx = max(ax, ay)
    var a = mn / (mx if mx > Float32(1e-10) else Float32(1e-10))
    var s = a * a
    var r = (Float32(-0.0464964749) * s + Float32(0.15931422)) * s
    r = (r - Float32(0.327622764)) * s * a + a
    if ay > ax: r = Float32(1.5707963267948966) - r
    if x < Float32(0.0): r = Float32(3.14159265358979323846) - r
    if y < Float32(0.0): r = -r
    return r


# ── SIMD math helpers (used in BVH and shading hot paths) ────────────────────

@always_inline
def cross(a: Vec3f, b: Vec3f) -> Vec3f:
    var a_yzx = Vec3f(a[1], a[2], a[0])
    var b_zxy = Vec3f(b[2], b[0], b[1])
    var a_zxy = Vec3f(a[2], a[0], a[1])
    var b_yzx = Vec3f(b[1], b[2], b[0])
    return a_yzx * b_zxy - a_zxy * b_yzx

@always_inline
def dot(a: Vec3f, b: Vec3f) -> Float32:
    var prod = a * b
    return prod[0] + prod[1] + prod[2]

# pbrt's two-sided reflection rule, as a frame choice. pbrt keeps the shading
# normal where the geometry puts it and makes every REFLECTION BxDF two-sided
# about it: DiffuseBxDF/ConductorBxDF return zero unless SameHemisphere(wo, wi)
# and use |cos|, LayeredBxDF (twoSided) mirrors wo/wi when wo.z < 0. Turning
# the shading normal toward wo is the same function with one line at the frame
# instead of a sign test in every BxDF.
#
# NOT for dielectrics: their normal must keep its raw orientation, because
# entering-vs-exiting is read from its sign (see shade_dielectric).
#
# What this replaced, and why it mattered: the shading sites used to fall back
# to the GEOMETRIC normal whenever a bump/normal map (or plain vertex-normal
# interpolation) tilted the shading normal past wo. At a grazing view about
# half a bump map's slopes do that, so half the relief was silently flattened:
# barcelona's displaced deck at ~4 degrees read 0.834 of pbrt, 1.002 with
# this rule.
@always_inline
def face_toward(n: Vec3f, w: Vec3f) -> Vec3f:
    if dot(n, w) < Float32(0.0):
        return -n
    return n

