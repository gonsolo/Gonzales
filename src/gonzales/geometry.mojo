from std.ffi import external_call
from std.memory import alloc
from std.math import sqrt, acos, atan2, cos, min, max, abs
from std.sys.info import align_of
from gonzales.spectrum import SampledWavelengths

# Value structs shared with GPU code can't hold Optional[UnsafePointer], so an
# "unset" pointer field is instead left at its `.unsafe_dangling()` sentinel --
# which Mojo returns as an address equal to align_of[T](), NOT a fixed value
# across pointee types (4 for Float32, 1 for UInt8, 8 for a pointer-sized
# struct, ...). Use this helper instead of a hand-picked magic-number
# threshold at each call site.
@always_inline
def _is_real_ptr[T: AnyType, O: Origin[mut=True]](ptr: UnsafePointer[T, O]) -> Bool:
    return Int(ptr) > align_of[T]()

# ── Math constants ─────────────────────────────────────────────────────────────
# Stored as Float32 aliases so every formula reads like the paper it came from.
# See: docs/01_geometry.md

# <<listing: Math Constants>>
comptime PI         : Float32 = 3.14159265358979323846
comptime TWO_PI     : Float32 = 6.28318530717958647692
comptime INV_PI     : Float32 = 0.31830988618379067154
comptime INV_TWO_PI : Float32 = 0.15915494309189533577
comptime INV_FOUR_PI: Float32 = 0.07957747154594766788
comptime SQRT2      : Float32 = 1.41421356237309504880
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
struct Vec3f(TrivialRegisterPassable):
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
def store_vec3[O: Origin[mut=True]](dst: UnsafePointer[Float32, O], slot: Int, v: Vec3f):
    """Write a Point3f/Vec3f (pass `.to_simd()`) into a flat, stride-3
    Float32 buffer at `slot` -- i.e. dst[slot*3 : slot*3+3] -- replacing the
    dst[slot*3+0]=v.x; dst[slot*3+1]=v.y; dst[slot*3+2]=v.z pattern repeated
    at per-pixel G-buffer-style write sites."""
    dst[slot*3+0] = v[0]
    dst[slot*3+1] = v[1]
    dst[slot*3+2] = v[2]

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
struct Bounds3f(TrivialRegisterPassable):
    """An axis-aligned bounding box (world space)."""
    var min: Point3f
    var max: Point3f

# ── Color / Spectrum ───────────────────────────────────────────────────────────
# SampledSpectrum is the RGB-typed running throughput/radiance accumulator
# threaded through PathState_C, BDPTVertex, SPPMPixel/SPPMPhoton, etc. It is
# deliberately RGB, not a 4-wide spectral type: real per-wavelength spectral
# evaluation (spectrum.mojo's SpectralSample, Jakob-Hanika upsampling) happens
# only at the specific points that need it -- NEE terms, medium transmittance
# -- converting back to RGB immediately so it can fold into this accumulator.
# Flipping this alias to a true spectral accumulator (full multi-bounce
# spectral coherence) is a distinct, much larger future project, not a
# mechanical rename -- see project_spectral_rendering memory's Stage 6 notes.
# See: docs/02_spectra_and_color.md

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

comptime SampledSpectrum = RGB
# <<listing: SampledSpectrum>>
"""RGB-typed running throughput/radiance accumulator. Intentionally == RGB,
   not a spectral type -- see the comment above this alias's declaration.
   See: docs/02_spectra_and_color.md
# <</listing>>
"""

# ── Scene primitives ───────────────────────────────────────────────────────────

@fieldwise_init
struct PrimId_C(TrivialRegisterPassable):
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
struct Instance_C(TrivialRegisterPassable):
    """One placement of a template (BLAS). `objToWorld`/`worldToObj` are 16-float
    column-major matrices (same convention as transform.mojo). A TLAS leaf of
    PrimId_C.type == 6 has id1 = index into SceneDescriptor2_C.instances.
    `blasIdx` indexes SceneDescriptor2_C.blasNodesArr/blasPrimIdsArr (one
    private BVH2 per template, each a separate allocation — no shared-pool
    offset arithmetic needed). A BLAS's PrimId_C entries use ordinary type==0
    triangle encoding against the same global `meshes` array as the TLAS."""
    var objToWorld: SIMD[DType.float32, 16]
    var worldToObj: SIMD[DType.float32, 16]
    var blasIdx:    Int32

struct MatKind:
    comptime diffuse           = Int8(1)
    comptime area_light        = Int8(2)
    comptime conductor         = Int8(3)
    comptime dielectric        = Int8(4)
    comptime coated_diffuse    = Int8(5)
    comptime diffuse_transmit  = Int8(6)
    comptime coated_conductor  = Int8(7)
    comptime mix               = Int8(8)
    comptime thin_dielectric   = Int8(9)
    comptime interface         = Int8(10)
    comptime hair              = Int8(11)
    comptime measured          = Int8(12)

@fieldwise_init
struct Material_C(TrivialRegisterPassable):
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var albedo: RGB
    var emission: RGB
    var tex_idx: Int32      # -1 = no texture; -2 = procedural checkerboard (see checker_* below); >= 0 = index into texture table
    var roughU: Float32     # GGX uroughness (conductor); 0 = perfect mirror
    var roughV: Float32     # GGX vroughness (conductor); 0 = perfect mirror
    var normal_tex_idx: Int32  # -1 = no normal map; >= 0 = index into texture table
    var rough_tex_idx: Int32   # -1 = no roughness map; >= 0 = index into texture table.
                                # Isotropic only (applies to both roughU/roughV) -- covers
                                # the "texture roughness" param, not separate per-axis
                                # "texture uroughness"/"texture vroughness" textures.
                                # Conductor only today (shade_conductor); not resolved for
                                # coated_conductor/dielectric/coated_diffuse's coat.
    var medium_interface_idx: Int32  # -1 = no medium interface bound
    # Procedural checkerboard params, valid only when tex_idx == -2.
    var checker_tex1: RGB
    var checker_tex2: RGB
    var checker_uscale: Float32
    var checker_vscale: Float32
    var measured_idx: Int32  # -1 = not a "measured" material; >= 0 = index into
                              # SceneDescriptor2_C.measuredBrdfs (see MeasuredBRDF_C)

# ── Measured (tabulated) BRDF ────────────────────────────────────────────────
# One instance per distinct ".bsdf" tensor file (deduped by path — a scene may
# reference the same file from many materials, e.g. sportscar's car-paint
# body). Holds the real Dupuy & Jakob PiecewiseLinear2D marginal/conditional
# distributions for vndf/luminance (with derived CDFs) plus the
# Evaluate-only ndf/sigma/spectra tables — see measured_bsdf.mojo's loader
# for construction and bxdf.mojo's bxdf_{eval,sample,pdf}_measured for the
# consumers. Same struct reused for both the CPU host-pointer instance
# (built by the loader) and the GPU device-pointer instance (built in
# gpu.mojo's scene upload, Stage 3) — mirrors TriangleMesh_C/Curve_C's
# existing host-then-device-pointer-reuse convention.
#
# CAUTION: this struct has many pointer fields, so it must NEVER be passed BY
# VALUE across a real (non-inlined) function-call boundary -- see
# spectrum.mojo's long comment on the confirmed Mojo compiler miscompilation
# class (modular/modular#6759). Always load a local `var mb = ...[idx]` and
# only hand it to @always_inline helpers.
@fieldwise_init
struct MeasuredBRDF_C(TrivialRegisterPassable):
    var isotropic:     Int32  # 1 if n_phi_i <= 2 (only isotropic supported today)
    var n_theta_i:     Int32
    var n_phi_i:       Int32
    var n_wavelengths: Int32
    var theta_i:     UnsafePointer[Float32, MutExternalOrigin]  # [n_theta_i]
    var phi_i:       UnsafePointer[Float32, MutExternalOrigin]  # [n_phi_i]
    var wavelengths: UnsafePointer[Float32, MutExternalOrigin]  # [n_wavelengths]

    # ndf / sigma: PiecewiseLinear2D<0> — Evaluate-only, no param axes, no CDF.
    var ndf_data:   UnsafePointer[Float32, MutExternalOrigin]  # [ndf_ys * ndf_xs]
    var ndf_xs:     Int32
    var ndf_ys:     Int32
    var sigma_data: UnsafePointer[Float32, MutExternalOrigin]  # [sigma_ys * sigma_xs]
    var sigma_xs:   Int32
    var sigma_ys:   Int32

    # vndf / luminance: PiecewiseLinear2D<2>, param axes (phi_i, theta_i),
    # both Sample+Evaluate-capable (marginal/conditional CDFs built). Both
    # share the same param resolution, hence the same stride2_* pair below.
    var vndf_data: UnsafePointer[Float32, MutExternalOrigin]  # [slices2 * vndf_ys * vndf_xs]
    var vndf_marg: UnsafePointer[Float32, MutExternalOrigin]  # [slices2 * vndf_ys]
    var vndf_cond: UnsafePointer[Float32, MutExternalOrigin]  # [slices2 * vndf_ys * vndf_xs]
    var vndf_xs:   Int32
    var vndf_ys:   Int32
    var lum_data: UnsafePointer[Float32, MutExternalOrigin]   # [slices2 * lum_ys * lum_xs]
    var lum_marg: UnsafePointer[Float32, MutExternalOrigin]   # [slices2 * lum_ys]
    var lum_cond: UnsafePointer[Float32, MutExternalOrigin]   # [slices2 * lum_ys * lum_xs]
    var lum_xs:   Int32
    var lum_ys:   Int32
    var stride2_phi:   Int32  # PiecewiseLinear2D<2>'s per-param-axis stride
    var stride2_theta: Int32  # (in units of one x*y slice); shared by vndf+luminance

    # spectra: PiecewiseLinear2D<3>, param axes (phi_i, theta_i, wavelengths),
    # Evaluate-only (no CDF) — its own stride triple (different Dimension
    # than vndf/luminance's, so the strides differ even though phi_i/theta_i
    # are shared).
    var spectra_data: UnsafePointer[Float32, MutExternalOrigin]  # [slices3 * spectra_ys * spectra_xs]
    var spectra_xs: Int32
    var spectra_ys: Int32
    var stride3_phi:    Int32
    var stride3_theta:  Int32
    var stride3_lambda: Int32

@fieldwise_init
struct TriangleMesh_C(TrivialRegisterPassable):
    var points: UnsafePointer[Float32, MutExternalOrigin]
    var faceIndices: UnsafePointer[Int64, MutExternalOrigin]
    var vertexIndices: UnsafePointer[Int64, MutExternalOrigin]
    var uvs: UnsafePointer[Float32, MutExternalOrigin]   # nullable; stride 2 floats per vertex
    var normals: UnsafePointer[Float32, MutExternalOrigin]  # nullable; stride 3 floats per vertex (shading normals)

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
    var pending_mat: Int8      # GPU only: MatKind of material awaiting per-material kernel (0 = none)
    var volume_scattered: Int8 # 1 if this bounce was a volume scatter; shade_nee_core skips miss handling
    # lastBsdfPdf: cosine-hemisphere PDF from the previous scatter (cos_theta / pi).
    # Used for MIS weighting when the next bounce hits an emitter.
    var lastBsdfPdf: Float32
    var current_medium_idx: Int32  # -1 = vacuum; >= 0 = index into scene.mediums
    var sampler_dim: Int32         # next Sobol dimension; 3 after primary ray (dims 0+1 film pos, dim 2 wavelength), +8 per bounce
    var sobol_idx: UInt64          # path's Z-Sobol sample index (from Morton code + si)
    # Hero-wavelength sample for spectral rendering (staged rollout, see
    # project_spectral_rendering memory / lovely-dazzling-meteor plan).
    # Sampled once per path at primary-ray generation (Sobol dim 2, see
    # gen_primary_ray_state); unused by shading until Stage 2b/2c wire
    # spectral BxDF/light evaluation through it.
    var wavelengths: SampledWavelengths
# <</listing>>
# PathState_C layout: 24+12+12+12+4+8+8+1+1+1+1+4+4+4+8+20 = 124 bytes (no trailing pad needed);
# size is computed via size_of[PathState_C]() everywhere (GPU buffer sizing included), not hardcoded.

# ── Lights ────────────────────────────────────────────────────────────────────
# See: docs/06_lights_and_materials.md

@fieldwise_init
struct AreaLight_C(TrivialRegisterPassable):
    """A sampleable area light. kind==0: a triangle mesh (meshIdx indexes
    TriangleMesh_C, n_tris triangles, total_area = mesh surface area).
    kind==1: a native curve (meshIdx reused as the curve's index into the
    scene's Curve_C array; n_tris unused; total_area = the curve's tube
    lateral surface area, see curve_light_tube_area). Both kinds are sampled
    uniformly-by-primitive (random triangle, or random curve piece) rather
    than area-weighted — see sample_area_light_uniform (sppm.mojo) and
    shading.mojo's NEE area-light sampling."""
    var meshIdx: Int32
    var n_tris: Int32       # number of triangles in this light mesh (kind==0 only)
    var emission: SampledSpectrum
    var total_area: Float32 # total surface area of this light mesh or curve tube
    var kind: Int8          # 0 = mesh triangle light, 1 = curve light
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8

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

@always_inline
def sphere_outward_normal(hit: Point3f, center: Point3f) -> Vec3f:
    """Normalized outward normal at a point on an analytic sphere's surface
    (normalize(hit - center)). Degenerate (hit == center) returns the zero
    vector, same fallback as every inline call site this replaces."""
    var n = hit - center
    var nl = n.length()
    return (n / nl) if nl > Float32(0.0) else n


# ── Native curve (hair/fur) primitive ──────────────────────────────────────────
# A single local cubic B-spline segment: 4 control points already windowed to
# this piece (mirrors PBRT's `Shape "curve"` per-segment convention). Stored
# and BVH-indexed natively — no triangle tessellation, no per-strand vertex
# buffers. Intersected lazily per ray as a short chain of locally-linear
# fixed-radius cylinders (see intersect_curve below).

@fieldwise_init
struct Curve_C(TrivialRegisterPassable):
    """One cubic B-spline curve segment. width0/width1 are the curve's full
    diameter at the segment's start/end (u=0/u=1 of THIS segment's local
    parameter range, not the whole original strand). n_pieces is a per-curve
    tessellation-at-trace-time hint (1..CURVE_N_PIECES), precomputed once at
    parse time from the control polygon's flatness — most real hair/fur
    exports chop each strand into thousands of already-near-straight
    directives, so forcing the full CURVE_N_PIECES chain on every one of
    them wastes GPU work and — because ray-adjacent pixels can pick very
    different piece counts — causes severe SIMT warp divergence."""
    var cp0: Point3f
    var cp1: Point3f
    var cp2: Point3f
    var cp3: Point3f
    var width0: Float32
    var width1: Float32
    var materialIndex: Int32
    var n_pieces: Int32

comptime CURVE_N_PIECES: Int = 7   # max locally-linear pieces per segment (8 samples)

# Divergence experiment (traverse_paths_gpu): max curve-leaf candidates recorded
# per ray before falling back to immediate intersection. See traverse_bvh2_core_defer_curves.
# The CUDA path's fallback (intersect_curve inline once over K) makes
# correctness independent of K there, so this only sizes the small,
# always-allocated curve_cand_prim_buf/curve_cand_offset_buf CUDA fallback
# buffers (gpu.mojo) -- kept modest since Vulkan-RT rendering doesn't use
# them at all anymore (see vulkaninterop_rt_create_scene's docstring): its
# own candidate pool is a SEPARATE, independently-sized Vulkan-interop
# buffer read directly by resolve_curve_candidates_gpu, with no CUDA-side
# duplicate to keep in sync -- see InteropRtScene::curveCandBuffer's
# comment (vulkaninterop.cpp) for that pool's own capacity constant.
comptime CURVE_DEFER_K: Int = 4

@always_inline
def _curve_perp_axis(t: Vec3f) -> Vec3f:
    """Deterministic unit vector perpendicular to t, built from t alone (no
    stored per-vertex data needed — reproducible at both intersect and shade time)."""
    var ux: Float32; var uy: Float32; var uz: Float32
    if abs(t[1]) < Float32(0.9):
        ux = -t[2]; uy = Float32(0.0); uz = t[0]
    else:
        ux = Float32(0.0); uy = t[2]; uz = -t[1]
    var v = Vec3f(ux, uy, uz)
    var l = sqrt(dot(v, v))
    if l > Float32(1e-12):
        return v * (Float32(1.0) / l)
    return Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))

@always_inline
def curve_bspline_point(curve: Curve_C, t: Float32) -> Vec3f:
    """Evaluate the uniform cubic B-spline segment at local parameter t in [0,1]."""
    var t2 = t * t
    var t3 = t2 * t
    var b0 = (Float32(1.0) - Float32(3.0)*t + Float32(3.0)*t2 - t3) / Float32(6.0)
    var b1 = (Float32(4.0) - Float32(6.0)*t2 + Float32(3.0)*t3) / Float32(6.0)
    var b2 = (Float32(1.0) + Float32(3.0)*t + Float32(3.0)*t2 - Float32(3.0)*t3) / Float32(6.0)
    var b3 = t3 / Float32(6.0)
    var p0 = curve.cp0.to_simd(); var p1 = curve.cp1.to_simd()
    var p2 = curve.cp2.to_simd(); var p3 = curve.cp3.to_simd()
    return p0*b0 + p1*b1 + p2*b2 + p3*b3

@always_inline
def curve_piece_endpoints(curve: Curve_C, piece: Int) -> Tuple[Vec3f, Vec3f, Float32, Float32]:
    """Sample points + radii bounding one of this curve's n_pieces locally-linear pieces."""
    var n = Float32(curve.n_pieces)
    var t0 = Float32(piece) / n
    var t1 = Float32(piece + 1) / n
    var q0 = curve_bspline_point(curve, t0)
    var q1 = curve_bspline_point(curve, t1)
    var r0 = (curve.width0 + (curve.width1 - curve.width0) * t0) * Float32(0.5)
    var r1 = (curve.width0 + (curve.width1 - curve.width0) * t1) * Float32(0.5)
    return (q0, q1, r0, r1)

@always_inline
def intersect_curve(
    ray_org: Vec3f,
    ray_dir: Vec3f,
    curve: Curve_C,
    first_piece: Int,
    piece_count: Int,
    tMax: Float32,
) -> Tuple[Bool, Float32, Float32, Float32]:
    """Ray vs. a small contiguous run of locally-linear pieces (fixed-radius
    cylinders) of a native curve segment: [first_piece, first_piece+piece_count).
    Returns (hit, t, h, v):
      h — signed fiber cross-section position in [-1,1] (PBRT HairBxDF convention)
      v — position along this curve segment's own [0,1] parameter range

    A BVH leaf covers one such run — see pbrt_parser.mojo's greedy
    piece-merging (adjacent pieces of the same curly curve get grouped into
    one leaf when they're still collinear enough that the group's bound
    isn't much looser than an individual piece's). piece_count is almost
    always 1-3 in practice (curls are locally gradual, not zigzag), only
    reaching curve.n_pieces in the rare case a whole curly segment is one
    flat run. This is what keeps the leaf count — and BVH build time — down
    without going back to one loose bound for the whole segment.

    Only the near root of the ray/infinite-cylinder quadratic is tested (not
    both roots) — deliberately simpler/leaner than a textbook capsule test.
    Adjacent pieces overlap at their shared endpoint, so a hit missed at one
    piece's far cap is picked up by the next piece's near-root test; the
    gap this could theoretically miss is far below pixel-visible scale given
    how short each piece already is.

    Measured @no_inline on an earlier version of this function: register
    count dropped only marginally and per-kernel GPU time got WORSE —
    call/return overhead outweighed any register win, so this stays
    @always_inline. Left as a note so this isn't retried blindly.
    """
    var n_pieces_f = Float32(curve.n_pieces)
    var best_hit = False
    var best_t = tMax
    var best_h = Float32(0.0)
    var best_v = Float32(0.0)

    for piece in range(first_piece, first_piece + piece_count):
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var r = (r0 + r1) * Float32(0.5)
        if r < Float32(1e-8):
            continue
        var axis = q1 - q0
        var axis_len_sq = dot(axis, axis)
        if axis_len_sq < Float32(1e-16):
            continue

        # Ray-vs-infinite-cylinder-around-`axis` via cross products — avoids
        # normalizing axis (no sqrt) for the common case where this piece
        # isn't hit at all; sqrt is only paid once a valid closer root exists.
        var oc = ray_org - q0
        var A = cross(axis, ray_dir)
        var B = cross(axis, oc)
        var a = dot(A, A)
        if a < Float32(1e-12):
            continue
        var b = Float32(2.0) * dot(A, B)
        var c = dot(B, B) - r*r*axis_len_sq
        var disc = b*b - Float32(4.0)*a*c
        if disc < Float32(0.0):
            continue

        var tc = (-b - sqrt(disc)) / (Float32(2.0) * a)
        if tc <= Float32(1e-4) or tc >= best_t:
            continue

        var axis_len = sqrt(axis_len_sq)
        var axis_dir = axis * (Float32(1.0) / axis_len)
        var hit_pos = (ray_org + tc*ray_dir) - q0
        var s = dot(hit_pos, axis_dir)
        if s < Float32(0.0) or s > axis_len:
            continue

        var offset_perp = hit_pos - s * axis_dir
        var u_axis = _curve_perp_axis(axis_dir)
        best_hit = True; best_t = tc
        best_h = max(Float32(-1.0), min(Float32(1.0), dot(offset_perp, u_axis) / r))
        best_v = (Float32(piece) + s/axis_len) / n_pieces_f

    return (best_hit, best_t, best_h, best_v)

@always_inline
def curve_piece_bounds(curve: Curve_C, piece: Int) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32]:
    """Tight AABB for one locally-linear piece: the segment q0..q1 thickened
    by max(r0,r1). Used as the BVH leaf bound now that leaves are per-piece
    instead of per-whole-curve (see intersect_curve)."""
    var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
    var r = max(r0, r1)
    var xmin = min(q0[0], q1[0]) - r
    var ymin = min(q0[1], q1[1]) - r
    var zmin = min(q0[2], q1[2]) - r
    var xmax = max(q0[0], q1[0]) + r
    var ymax = max(q0[1], q1[1]) + r
    var zmax = max(q0[2], q1[2]) + r
    return (xmin, ymin, zmin, xmax, ymax, zmax)

@always_inline
def curve_light_tube_area(curve: Curve_C) -> Float32:
    """Approximate lateral (side) surface area of a curve treated as a thin
    tube light: sum over each locally-linear piece of its lateral cylinder
    area (2*pi*avg_radius*piece_length). Used only to normalize NEE/light-
    sampler PDFs for emissive curves (AreaLight_C.kind==1) against the
    uniform-random-piece sampling used at runtime — see
    sample_area_light_uniform (sppm.mojo) and shading.mojo's curve NEE
    branch, which both pick pieces uniformly rather than area-weighted
    (mirroring the pre-existing mesh-light approximation of picking a
    uniform random triangle rather than area-weighting by triangle)."""
    var area = Float32(0.0)
    for piece in range(Int(curve.n_pieces)):
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var seg = q1 - q0
        var seg_len = sqrt(dot(seg, seg))
        area += Float32(2.0) * Float32(3.14159265358979323846) * ((r0 + r1) * Float32(0.5)) * seg_len
    return area


# ── Homogeneous / heterogeneous media ─────────────────────────────────────────

@fieldwise_init
struct Medium_C(TrivialRegisterPassable):
    """Participating medium (PBRT-v4 HomogeneousMedium / GridMedium "uniformgrid").
    sigma_a + sigma_s are pre-scaled by 'scale'. For a heterogeneous medium
    (grid_idx >= 0), sigma_a/sigma_s are the PER-UNIT-DENSITY coefficients —
    actual extinction at a point is density(point) * (sigma_a + sigma_s), see
    Grid_C/grid_sample_density below.
    g = Henyey-Greenstein anisotropy in [-1, 1]; 0 = isotropic.
    """
    var sigma_a: SampledSpectrum   # absorption coefficient (1/m), per unit density if grid_idx >= 0
    var sigma_s: SampledSpectrum   # scattering coefficient (1/m), per unit density if grid_idx >= 0
    var g:       Float32           # HG anisotropy
    var grid_idx: Int32            # -1 = homogeneous; >=0 = index into scene.grids
    var _pad1:   Float32
    var _pad2:   Float32

@fieldwise_init
struct Grid_C(TrivialRegisterPassable):
    """Dense heterogeneous density grid backing a Medium_C (PBRT-v4
    "uniformgrid" — a flat array of nx*ny*nz density samples spanning the
    axis-aligned box [p0,p1] in the medium's own local space). density(x) is
    a unitless multiplier: real sigma_t at a world point = grid_sample_density(x)
    * (medium.sigma_a + medium.sigma_s). world_to_medium maps a world-space
    point into that local [p0,p1]-space box (same 4x4 column-major convention
    as transform.mojo's transform_points: m[0],m[4],m[8],m[12] combine for x).
    max_density is the majorant used for delta-tracking free-flight sampling.
    """
    var density: UnsafePointer[Float32, MutExternalOrigin]  # flat, nz-major: idx = (z*ny + y)*nx + x
    var nx: Int32
    var ny: Int32
    var nz: Int32
    var p0: Point3f
    var p1: Point3f
    var world_to_medium: SIMD[DType.float32, 16]
    var max_density: Float32

@always_inline
def _grid_density_at(grid: Grid_C, xi: Int, yi: Int, zi: Int) -> Float32:
    var cx = max(0, min(Int(grid.nx) - 1, xi))
    var cy = max(0, min(Int(grid.ny) - 1, yi))
    var cz = max(0, min(Int(grid.nz) - 1, zi))
    return grid.density[(cz * Int(grid.ny) + cy) * Int(grid.nx) + cx]

@always_inline
def grid_sample_density(grid: Grid_C, p_world: Vec3f) -> Float32:
    """Trilinearly-interpolated density at a world-space point; 0 outside
    the grid's local bounds [p0,p1]."""
    var m = grid.world_to_medium
    var px = m[0]*p_world[0] + m[4]*p_world[1] + m[8]*p_world[2] + m[12]
    var py = m[1]*p_world[0] + m[5]*p_world[1] + m[9]*p_world[2] + m[13]
    var pz = m[2]*p_world[0] + m[6]*p_world[1] + m[10]*p_world[2] + m[14]

    var ext_x = grid.p1.x - grid.p0.x
    var ext_y = grid.p1.y - grid.p0.y
    var ext_z = grid.p1.z - grid.p0.z
    if ext_x <= Float32(0.0) or ext_y <= Float32(0.0) or ext_z <= Float32(0.0):
        return Float32(0.0)
    var u = (px - grid.p0.x) / ext_x
    var v = (py - grid.p0.y) / ext_y
    var w = (pz - grid.p0.z) / ext_z
    if u < Float32(0.0) or u > Float32(1.0) or v < Float32(0.0) or v > Float32(1.0) or w < Float32(0.0) or w > Float32(1.0):
        return Float32(0.0)

    # Continuous voxel coords (PBRT convention: sample centers at half-integer
    # offsets, so u=0..1 maps to [-0.5, nx-0.5]).
    var gx = u * Float32(grid.nx) - Float32(0.5)
    var gy = v * Float32(grid.ny) - Float32(0.5)
    var gz = w * Float32(grid.nz) - Float32(0.5)
    var x0 = Int(gx); var y0 = Int(gy); var z0 = Int(gz)
    var fx = gx - Float32(x0); var fy = gy - Float32(y0); var fz = gz - Float32(z0)
    if gx < Float32(0.0): x0 -= 1; fx = gx - Float32(x0)
    if gy < Float32(0.0): y0 -= 1; fy = gy - Float32(y0)
    if gz < Float32(0.0): z0 -= 1; fz = gz - Float32(z0)

    var d00 = _grid_density_at(grid, x0, y0, z0)     * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0, z0)     * fx
    var d10 = _grid_density_at(grid, x0, y0+1, z0)   * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0+1, z0)   * fx
    var d01 = _grid_density_at(grid, x0, y0, z0+1)   * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0, z0+1)   * fx
    var d11 = _grid_density_at(grid, x0, y0+1, z0+1) * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0+1, z0+1) * fx
    var d0 = d00 * (Float32(1.0) - fy) + d10 * fy
    var d1 = d01 * (Float32(1.0) - fy) + d11 * fy
    return d0 * (Float32(1.0) - fz) + d1 * fz

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
    var cdf_ptr: UnsafePointer[Float32, MutExternalOrigin]   # flat 2D CDF (marginal + conditional)
    var pixels_ptr: UnsafePointer[Float32, MutExternalOrigin] # raw HDR pixels, 3 floats/pixel (CPU only)
    var world_to_light: UnsafePointer[Float32, MutExternalOrigin]  # 16-float col-major inverse of light CTM

# ── GPU / render pipeline helpers ─────────────────────────────────────────────

@fieldwise_init
struct GpuTexture_C(TrivialRegisterPassable):
    var data: UnsafePointer[Float32, MutExternalOrigin]  # device pointer: full mip pyramid, contiguous, pre-linearised float RGB
    var width: Int32                                 # level-0 width
    var height: Int32                                # level-0 height
    var n_levels: Int32                              # number of mip levels stored in `data` (>=1)

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
struct LightSampler_C(TrivialRegisterPassable):
    """Power-weighted CDF over area lights.
    cdf[0]=0, cdf[n]=1; pdf[i] = cdf[i+1] - cdf[i] = power_i / total_power.
    Built at parse time; on GPU the cdf pointer is patched to device memory.
    """
    var cdf: UnsafePointer[Float32, MutExternalOrigin]  # n+1 entries
    var n: Int32
    var _pad: Int32

@always_inline
def light_sampler_sample(ls: LightSampler_C, u: Float32) -> Tuple[Int, Float32]:
    """Binary-search the CDF. Returns (light_index, selection_pdf)."""
    var lo = 0
    var hi = Int(ls.n) - 1
    while lo < hi:
        var mid = (lo + hi) >> 1
        if ls.cdf[mid + 1] <= u:
            lo = mid + 1
        else:
            hi = mid
    var pdf = ls.cdf[lo + 1] - ls.cdf[lo]
    return (lo, max(pdf, Float32(1e-6)))

@always_inline
def light_sampler_pdf(ls: LightSampler_C, light_idx: Int32) -> Float32:
    """Selection pdf for a KNOWN light index -- the inverse of
    light_sampler_sample's (index, pdf) draw, needed to re-evaluate a
    specific light's pdf without redrawing it (e.g. ReSTIR DI's MIS weight
    for an already-chosen reservoir winner, restir_di.mojo)."""
    var i = Int(light_idx)
    return max(ls.cdf[i + 1] - ls.cdf[i], Float32(1e-6))


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
# <<listing: schlick_fresnel>>

@always_inline
def schlick_fresnel(cos_theta: Float32, f0: Float32) -> Float32:
    """Schlick (1994) approximation to the Fresnel reflectance.
    f0 = ((η−1)/(η+1))² is the normal-incidence reflectance.
    See: docs/05_reflection_models.md — Fresnel.
    """
    var t = Float32(1.0) - cos_theta
    var t2 = t * t
    return f0 + (Float32(1.0) - f0) * (t2 * t2 * t)
# <</listing>>

@always_inline
def fr_dielectric(cos_theta_i_in: Float32, eta_in: Float32) -> Float32:
    """Exact unpolarized Fresnel reflectance for a dielectric interface.
    eta = η_t / η_i (relative IOR). Returns 1.0 on total internal reflection.
    Handles both sides: a negative cos_theta_i means the ray arrives from the
    transmitted side, so the interface is flipped. Mirrors PBRT's FrDielectric.
    See: docs/05_reflection_models.md — Fresnel.
    """
    var cos_theta_i = max(Float32(-1.0), min(Float32(1.0), cos_theta_i_in))
    var eta = eta_in
    if cos_theta_i < Float32(0.0):
        eta = Float32(1.0) / eta
        cos_theta_i = -cos_theta_i
    var sin2_theta_i = max(Float32(0.0), Float32(1.0) - cos_theta_i * cos_theta_i)
    var sin2_theta_t = sin2_theta_i / (eta * eta)
    if sin2_theta_t >= Float32(1.0):
        return Float32(1.0)   # total internal reflection
    var cos_theta_t = safe_sqrt(Float32(1.0) - sin2_theta_t)
    var r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t)
    var r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t)
    return (r_parl * r_parl + r_perp * r_perp) * Float32(0.5)

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
def spherical_theta(v: Vec3f) -> Float32:
    """Polar angle θ ∈ [0, π] of a unit vector (y = up convention)."""
    return acos(max(Float32(-1.0), min(Float32(1.0), v.y)))

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

@always_inline
def spherical_phi(v: Vec3f) -> Float32:
    """Azimuthal angle φ ∈ [0, 2π] of a unit vector (y = up convention)."""
    var p = _atan2f(v.z, v.x)
    return p if p >= Float32(0.0) else p + TWO_PI

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
