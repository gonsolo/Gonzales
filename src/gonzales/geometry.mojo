from std.ffi import external_call
from std.memory import alloc
from std.math import sqrt, acos, atan2, cos, sin, min, max, abs, floor, log, exp
from std.sys.info import align_of
from gonzales.spectrum import SampledWavelengths, SpectralSample, spec_refl_unbounded
from gonzales.nanovdb import nvdb_sample_index, nvdb_majorant_at, nvdb_leaf_base, nvdb_leaf_value
from gonzales.rng import PCG32

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
struct FilmDims(TrivialRegisterPassable):
    """Film/framebuffer resolution (width, height), replacing the separate
    fw/fh pair on GpuSceneHandle and gpu_upload_scene. GPU kernels still take
    fw_dp/fh_dp as separate scalars: this struct doesn't implement
    DevicePassable, so it can't be an enqueue_function argument as-is."""
    var width: Int32
    var height: Int32

@fieldwise_init
struct FilterParams(TrivialRegisterPassable):
    """Pixel-reconstruction filter parameters, grouped on GpuSceneHandle and
    gpu_upload_scene; unpacked into scalars at kernel launches (see FilmDims)."""
    var sigma: Float32
    var support_x: Float32
    var support_y: Float32
    var norm_x: Float32
    var norm_y: Float32
    var type: Int32

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
    # 1 when this material is the BOUNDARY of a subsurface interior (its
    # medium interface's inside medium has Medium_C.is_sss). Set by
    # pbrt_parser's medium-interface binding pass. Boundary interactions on
    # such a surface -- entry, exit, and especially total internal reflection
    # -- must NOT be charged to the scene's maxdepth: the whole
    # enter/walk/exit sequence models ONE BSSRDF scattering event, exactly
    # the reasoning that already exempts the interior walk steps via
    # Medium_C.is_sss. Charging them silently ate light trapped by TIR (a
    # white furnace lost 10.7% at eta 1.5 and 30% at eta 2.0); see
    # Scenes/sss_furnace_sweep.py. Occupies a former padding byte, so
    # Material_C's size and every existing constructor call site are
    # unchanged.
    var sss_boundary: Int8
    var _pad1: Int8
    var _pad2: Int8
    var albedo: RGB
    var emission: RGB
    var tex_idx: Int32      # -1 = no texture; -2 = procedural checkerboard (see checker_* below); >= 0 = index into texture table
    var roughU: Float32     # GGX uroughness (conductor); 0 = perfect mirror
    var roughV: Float32     # GGX vroughness (conductor); 0 = perfect mirror
    var normal_tex_idx: Int32  # -1 = no normal map; >= 0 = index into texture table
    var bump_tex_idx: Int32    # -1 = no bump/displacement map; >= 0 = index into texture table
    var bump_scale: Float32    # height multiplier applied to bump_tex_idx's raw [0,1] value
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
    # Affine correction applied to tex_idx's looked-up value in shading.mojo's
    # _tex_lookup: `albedo = tex_bias + tex_scale * texture(uv)`, per channel.
    # Identity is scale=1, bias=0. This one form covers every texture-graph
    # shape the corpus actually uses on reflectance:
    #   "scale" (imagemap * s)      -> scale = s,          bias = 0
    #   "mix" (texture, const, a)   -> scale = 1 - a,      bias = a * const
    #   "mix" (const, texture, a)   -> scale = a,          bias = (1-a) * const
    #   "mix" (c1, c2, texture amt) -> scale = c2 - c1,    bias = c1
    # and composes under nesting, since an affine function of an affine
    # function is affine. Resolved by material_builder.mojo's
    # _resolve_affine_rgb; shapes needing two independent texture lookups
    # (a texture-times-texture product) are not representable and warn there.
    var tex_scale: RGB
    var tex_bias:  RGB

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
# spectrum.mojo's long comment on the suspected Mojo miscompilation class
# (modular/modular#6759, later retracted by its own author as
# unreproducible; kept as a defensive workaround regardless). Always load a
# local `var mb = ...[idx]` and only hand it to @always_inline helpers.
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
    # Path TRANSPORT is spectral: these carry radiance/weight at this path's
    # own 4 hero wavelengths, and RGB appears only at the boundaries (light
    # and material lookup on the way in, film splat on the way out).
    var throughput: SpectralSample
    var estimate: SpectralSample
    var albedo: RGB      # denoiser AOV -- an output, stays RGB
    var bounce: Int32
    var pcgState: UInt64
    var pcgInc: UInt64
    var active: Int8
    var specularBounce: Int8   # 1 if previous scatter was a delta BSDF (mirror/glass)
    var pending_mat: Int8      # GPU only: MatKind of material awaiting per-material kernel (0 = none)
    var volume_scattered: Int8 # 1 if this bounce was a volume scatter; shade_nee_core skips miss handling
    # 1 once this path has used its full `maxdepth` budget of REAL scattering
    # events. It is NOT dead yet: pbrt checks `depth++ >= maxDepth` at a
    # scatter BEFORE that vertex's NEE and before sampling a new direction, so
    # the segment LEAVING the last allowed vertex is still traced and whatever
    # emitter it lands on is still collected. A capped path therefore gets one
    # more intersect, may cross null interfaces, and may pick up emission --
    # but must not do NEE and must not scatter again.
    var at_cap: Int8
    # 1 if the last NON-specular vertex delegates its specular-chain lighting
    # to MNEE/SMS. That strategy samples exactly the paths
    # diffuse -> (specular chain) -> emitter, so when the path then REACHES
    # an emitter through a specular chain, the BSDF-sampling strategy is
    # reproducing a path already counted and its emission must be dropped.
    # Without this the two are simply summed: on the SMS caustic scene that
    # doubled the caustic (0.44 against a path-traced 0.234, with each
    # strategy contributing ~0.23 on its own). The reference integrator does
    # the same thing (path_sms_ss.cpp: it suppresses the emitter hit when
    # the previous vertex was a caustic receiver).
    var sms_covered: Int8
    # World position of that vertex, so the emitter-hit site can re-ask the
    # SAME question MNEE asked -- "is there glass on the straight segment
    # between this vertex and that point on the emitter?" -- for the point
    # the BSDF ray actually landed on.
    var last_ns_p: Vec3f
    # lastBsdfPdf: cosine-hemisphere PDF from the previous scatter (cos_theta / pi).
    # Used for MIS weighting when the next bounce hits an emitter.
    var lastBsdfPdf: Float32
    var current_medium_idx: Int32  # -1 = vacuum; >= 0 = index into scene.mediums
    # IOR of the dielectric SURFACE the path is currently transmitted inside
    # of; 1.0 = vacuum. bxdf_sample_dielectric's entering/exiting test is
    # purely local per-surface and otherwise always assumes the far side of
    # any interface is vacuum -- correct for an isolated pane, wrong for two
    # touching dielectric surfaces with no air gap (dense CAD assemblies,
    # nested/adjoining parts): a ray already inside glass A that
    # enters touching glass B got eta = 1/ior_B (as if from vacuum) instead
    # of ior_A/ior_B. For A==B (an optically invisible seam) that measured
    # 0.904x on Scenes/dielectric-touching-same-ior.pbrt where it should read
    # 1.0. Updated only on a TRANSMIT (not reflect/TIR, which stays in
    # whatever medium the path already occupied), on both ENTERING (new
    # value = this surface's ior) and EXITING (new value = the ONE saved
    # `previous_dielectric_ior`, see below -- NOT unconditionally vacuum
    # anymore).
    var current_dielectric_ior: Float32
    # The medium current_dielectric_ior held BEFORE the most recent ENTERING
    # transmit -- i.e. what to restore on the matching exit. Together the two
    # fields make a depth-2 stack (current + one level below), pushed on
    # enter and popped on exit, instead of current_dielectric_ior alone
    # (which has no memory at all once it's overwritten).
    #
    # Why this matters beyond the touching-seam case current_dielectric_ior
    # alone already fixed: for real multi-part CAD assemblies (see
    # transparent-machines), a ray entering touching glass B while still
    # inside glass A correctly reads eta=1 there (current_dielectric_ior
    # alone handles it) -- but the OLD exiting formula (`eta = ior`,
    # unconditional) then computes the EXIT out of B back into A as if B's
    # far side were vacuum too. For A==B that wrongly inflates eta from 1
    # back up to ior, and at grazing angles that spuriously triggers TOTAL
    # INTERNAL REFLECTION on a boundary that is physically invisible --
    # trapping the ray in a runaway TIR cascade through the assembly's many
    # touching interfaces instead of letting it escape. Traced directly via
    # `--pixel` on transparent-machines: a ray legitimately transmits
    # mesh124->mesh126 (same BK7 material, eta=1, fresnel=0, exactly
    # correct) and then immediately TIRs exiting mesh126 back toward
    # mesh124 (eta wrongly computed as 1.5196 instead of 1.0), continuing to
    # TIR for the rest of the 8-bounce debug trace without escaping -- the
    # direct mechanism behind the "flat, uniformly dark, missing highlights"
    # look, not noise or Russian roulette.
    #
    # Pushed on ENTERING (previous <- old current, before current is
    # overwritten to the new surface's ior) and popped on EXITING (current
    # <- previous, then previous resets to vacuum -- this is still only
    # depth 2: a THIRD level of nesting loses the level below B once B is
    # exited, exactly the same scoped, documented single-level-of-memory
    # limitation as before, just one level deeper than a bare scalar can
    # reach). Unchanged on reflect/TIR, same as current_dielectric_ior.
    var previous_dielectric_ior: Float32
    # Accumulated (eta_i/eta_t)^2 product across every dielectric TRANSMIT
    # this path has made so far; 1.0 = none yet. Mirrors pbrt-v4's
    # PathIntegrator::Li `etaScale` (cpu/integrators.cpp). Exists ONLY to
    # correct Russian roulette's termination probability, never applied to
    # the real throughput: bxdf_sample_dielectric's radiance_transmit factor
    # (eta*eta) temporarily compresses `throughput` while a path is INSIDE a
    # higher-index medium (entering eta<1 shrinks it) -- correct and exactly
    # cancelled once the path exits back out (eta>1 restores it) -- but
    # `_apply_russian_roulette` reads raw throughput luminance, so it sees
    # this mid-transit dip as if it were real attenuation and kills paths
    # that were about to be restored to full brightness on exit. For a
    # multi-bounce TIR chain through a complex glass assembly (see
    # transparent-machines: object-region highlights read as sparse
    # firefly-clamped spikes over an otherwise uniformly darker body instead
    # of pbrt's smooth bright highlight fan-out -- exactly the signature of
    # RR prematurely killing long, temporarily-dim, ultimately-bright paths,
    # with the rare survivors over-boosted by RR's own 1/(1-q) compensation)
    # this compounds badly. `eta_scale` divides the mid-transit compression
    # back out for the RR decision only, matching pbrt's `rrBeta = beta *
    # etaScale`. Updated only on a genuine TRANSMIT (reflect/TIR leave it
    # unchanged, same condition as current_dielectric_ior above).
    var eta_scale: Float32
    var sampler_dim: Int32         # next Sobol dimension; 3 after primary ray (dims 0+1 film pos, dim 2 wavelength), +8 per bounce
    var sobol_idx: UInt64          # path's Z-Sobol sample index (from Morton code + si)
    # Hero-wavelength sample for spectral rendering (staged rollout, see
    # project_spectral_rendering memory / lovely-dazzling-meteor plan).
    # Sampled once per path at primary-ray generation (Sobol dim 2, see
    # gen_primary_ray_state); unused by shading until Stage 2b/2c wire
    # spectral BxDF/light evaluation through it.
    var wavelengths: SampledWavelengths
    # Distance this path has travelled through NULL INTERFACES since its last
    # REAL scattering event.
    #
    # The emitter-hit MIS weight needs pdf_light in solid angle AT THE VERTEX
    # WHOSE BSDF/PHASE SAMPLE GENERATED THIS DIRECTION, because that is the
    # vertex where the competing NEE strategy was evaluated. It used to take
    # that distance as `inter.tHit` -- but shade_interface advances the ray
    # ORIGIN to the boundary it just crossed, so for any path that scattered
    # inside a medium and then left it, inter.tHit measures from the boundary,
    # not from the scattering vertex.
    #
    # With a light close to the medium's boundary the two differ enormously:
    # boundary->light ~0.05 gives pdf_light ~0.0025 and an emitter-hit MIS
    # weight of ~0.999, while the true scatter->light ~2 gives pdf_light ~4
    # and a weight of ~0.001. Both NEE and BSDF sampling then took nearly FULL
    # weight instead of partitioning, and the two summed: measured on an
    # area-lit slab, NEE alone 0.861x pbrt, phase sampling alone 0.975x,
    # combined 1.80x ~= their sum. Adding this back to inter.tHit restores the
    # real distance. A null interface never bends the ray, so accumulating
    # scalar distance is exact, however many boundaries are crossed.
    var mis_null_dist: Float32
# <</listing>>
# PathState_C layout: 24+12+12+12+4+8+8+1+1+1+1+4+4+4+4+4+4+4+8+20+4 = 140 bytes (was 136 -- +4 for previous_dielectric_ior);
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
    var emission: RGB
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
    var emission: RGB

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
    var sigma_a: RGB   # absorption coefficient (1/m), per unit density if grid_idx >= 0
    var sigma_s: RGB   # scattering coefficient (1/m), per unit density if grid_idx >= 0
    var g:       Float32           # HG anisotropy
    var grid_idx: Int32            # -1 = homogeneous; >=0 = index into scene.grids (dense "uniformgrid")
    var nvdb_idx: Int32            # -1 = none; >=0 = index into scene.nvdb_grids (sparse "nanovdb")
    # Emissive volumes (pbrt NanoVDBMedium): a SECOND nanovdb grid, named
    # "temperature", read from the same file and stored as an ordinary entry in
    # the same scene.nvdb_grids array -- so it reuses NvdbGrid_C, its upload,
    # and nvdb_sample_density unchanged. -1 = not emissive. Emitted radiance at
    # a point follows pbrt exactly: temp = (grid(p) - temp_offset) *
    # temp_scale, no emission at or below 100 K, then
    # Le = le_scale * blackbody(temp).
    var nvdb_temp_idx: Int32
    var le_scale:    Float32
    var temp_offset: Float32
    var temp_scale:  Float32
    # 1 if this medium is the INTERIOR of a `Material "subsurface"` object
    # (see material_builder.mojo). Subsurface scattering is rendered as a
    # plain random walk through this medium behind a dielectric boundary, so
    # the only thing that distinguishes it from an ordinary participating
    # medium is bookkeeping: its scattering events are interior random-walk
    # steps, not path bounces, and must NOT be charged to the path's
    # maxdepth budget. Skin1 at the scale sssdragon uses is ~37-50 extinction
    # events per scene unit with a red-channel albedo of 0.996, so a walk
    # routinely takes tens to hundreds of steps -- against pbrt's default
    # maxdepth of 5 the object would render nearly black. pbrt has the same
    # property for a different reason: its BSSRDF resolves the whole interior
    # analytically and never spends path depth on it either.
    var is_sss:      Int32


# ── Homogeneous-medium free-flight sampling ───────────────────────────────────
# Shared by BDPT, SPPM and the plain path tracer (CPU + GPU, gpu.mojo's
# _sample_medium_core) for the achromatic-decision, homogeneous case (glass-
# of-water / volumetric-caustic scenes) — no delta-tracking needed. Lives here
# (not in a higher-level integrator file) precisely so gpu.mojo can reach it
# too: geometry.mojo already sits below every integrator module and already
# imports spectrum.mojo (for SampledWavelengths/SpectralSample) and defines
# Medium_C/RGB, so it is the one place all three consumers can import from
# without a circular dependency (sppm.mojo imports gpu.mojo, so gpu.mojo can
# never import sppm.mojo's functions directly).
#
# Moved here 2026-09-10 from sppm.mojo (project_elegance_backlog_2026_09_10
# item 1) as part of collapsing three historically independent
# implementations of this same physical operation down to a smaller shared
# core. Full unification turned out to be a partial win, not a total one:
# BDPT and SPPM were ALREADY calling this exact function (bdpt.mojo imports
# it from sppm.mojo) by the time this pass started, so only the free-flight-
# DISTANCE sampling and the chromatic transmittance-RATIO math needed
# extracting for gpu.mojo to share too (see medium_transmittance_ratio_spectral
# below, and _sample_medium_core's own call site). gpu.mojo's homogeneous
# branch keeps its OWN scatter/absorb decision: it plays a physical
# Russian-roulette coin against sigma_s.r/sigma_t.r and terminates the path
# outright on absorption, whereas this struct's `weight`/`albedo` fields
# instead let BDPT/SPPM continue deterministically with throughput scaled by
# the exact albedo — two different, individually unbiased estimators of the
# same integral, not two buggy copies of one, so they were deliberately left
# separate rather than forced through one return shape. See
# docs/09_volumetric_media.md, "One shared free-flight core".
@fieldwise_init
struct HomogeneousFreeFlight(TrivialRegisterPassable):
    """Result of sampling a free-flight distance through a homogeneous medium
    by its red/hero-wavelength extinction coefficient (the same "sample by
    one channel, let the rest cancel analytically" convention used
    throughout gonzales's spectral MIS)."""
    var collided: Bool
    var t_free: Float32     # sampled distance (meaningful either way)
    var sig_t:   Float32    # red-channel extinction sigma_a.r + sigma_s.r
    var albedo:  RGB        # single-scattering albedo at the collision point (only if collided)
    var weight:  RGB        # per-channel chromatic WEIGHT, meaningful in BOTH branches.
                            # A pure RATIO to the sampled (red) channel, never a raw
                            # Beer-Lambert factor -- exactly 1 on red, and exactly 1 on
                            # every channel for a grey medium. Callers must multiply it
                            # into beta/flux in BOTH branches.

@always_inline
def sample_homogeneous_free_flight(med: Medium_C, t_surf: Float32, mut pcg: PCG32) -> HomogeneousFreeFlight:
    var sigma_t = med.sigma_a + med.sigma_s
    var sig_t = sigma_t.r
    if sig_t <= Float32(0.0):
        return HomogeneousFreeFlight(False, t_surf, sig_t, RGB(Float32(0)), RGB(Float32(1)))
    var t_free = -log(max(pcg.next_float(), Float32(1e-7))) / sig_t
    if t_free < t_surf:
        var alb_s = med.sigma_s.r / sig_t
        var alb_g_s = med.sigma_s.g / sigma_t.g if sigma_t.g > Float32(0.0) else alb_s
        var alb_b_s = med.sigma_s.b / sigma_t.b if sigma_t.b > Float32(0.0) else alb_s
        # COLLISION WEIGHT. The distance was sampled from red's exponential,
        # pdf(t) = sigma_t.r * exp(-sigma_t.r * t), so lane c's contribution
        # sigma_s_c * exp(-sigma_t_c * t) needs
        #     sigma_s_c*exp(-sigma_t_c*t) / (sigma_t.r*exp(-sigma_t.r*t))
        # and `albedo` above already carries sigma_s_c/sigma_t_c, leaving
        #     exp(-(sigma_t_c - sigma_t.r)*t) * sigma_t_c/sigma_t.r.
        # (Algebraically identical to gpu.mojo's albedo_r * sigma_s_c/sigma_s.r
        # form, just factored to keep `albedo` per-channel here.)
        #
        # This weight was MISSING entirely: the collision branch returned a
        # flat RGB(1), so VCM/SPPM were chromatically BIASED in any medium
        # whose channels differ -- a red-heavy medium lost exactly the green
        # and blue extinction ratio at every scattering event, compounding
        # per bounce. gpu.mojo's _sample_medium_core has always applied it
        # (its comment spells out both factors); it simply never propagated
        # to this shared BDPT/SPPM sampler -- the same fixed-in-one-consumer
        # split that accounts for most defects in this codebase. Exactly 1 on
        # every channel for a grey medium, so grey renders are unaffected.
        var wg = (exp(-(sigma_t.g - sigma_t.r) * t_free) * sigma_t.g / sig_t
                  if sigma_t.g > Float32(0.0) else Float32(1.0))
        var wb = (exp(-(sigma_t.b - sigma_t.r) * t_free) * sigma_t.b / sig_t
                  if sigma_t.b > Float32(0.0) else Float32(1.0))
        return HomogeneousFreeFlight(True, t_free, sig_t,
                                     RGB(alb_s, alb_g_s, alb_b_s),
                                     RGB(Float32(1.0), wg, wb))
    # Pass-through WEIGHT, not the raw Beer-Lambert factor. The distance was
    # sampled from the red channel, so P(reach the surface) is ALREADY
    # exp(-sigma_t.r * t_surf) -- that channel's transmittance is carried by
    # the sampling probability itself. Only the RATIO of each channel's
    # transmittance to the sampled one survives as a weight:
    #     exp(-sigma_t_c * t) / exp(-sigma_t.r * t) = exp(-(sigma_t_c - sigma_t.r) * t)
    # which is exactly 1 on red, and exactly 1 on every channel for a grey
    # medium.
    #
    # This used to return the FULL exp(-sigma_t_c * t) on all three channels,
    # which callers multiply into beta/flux -- double-counting the sampled
    # channel's transmittance, so the expected contribution of a surface seen
    # through a medium was exp(-2*sigma_r*t) where it should be
    # exp(-sigma_r*t). Surfaces behind a medium came out too dark by exactly
    # the transmittance, and unlike the chromatic terms this bit GREY media
    # too. gpu.mojo's _sample_medium_core has always used the ratio (its own
    # comment records the same fix); it simply never propagated to this
    # BDPT/SPPM sampler.
    var t_ref = exp(-sigma_t.r * t_surf)
    if t_ref < Float32(1e-30): t_ref = Float32(1e-30)
    var Tr = RGB(Float32(1.0),
                 exp(-sigma_t.g * t_surf) / t_ref,
                 exp(-sigma_t.b * t_surf) / t_ref)
    return HomogeneousFreeFlight(False, t_free, sig_t, RGB(Float32(0)), Tr)


# ── Genuinely spectral free-flight weight (chromatic media) ─────────────────
# `HomogeneousFreeFlight.weight` above is a RATIO computed entirely in RGB
# (exp(-sigma_t.g*t), exp(-sigma_t.b*t), ...) and every caller used to lift it
# into the 4 hero lanes by BAND-PICKING that already-exponentiated triple --
# i.e. selecting, per lane, which of the 3 RGB ratios to read. Band-picking a
# genuine coefficient (see rgb_bands_to_spectral_sample's own docstring, "a
# coefficient is not a colour") is fine on its own, but here it throws away
# information for free: an extinction coefficient IS spectral data, and
# gonzales already has a real RGB->spectrum upsampler for exactly this shape
# of quantity (spec_refl_unbounded, used elsewhere for reflectance-shaped
# weights that may exceed 1 -- an extinction coefficient is the same kind of
# thing). The two orders are NOT interchangeable for a smooth upsampler:
# upsample(exp(-sigma*t)) != exp(-upsample(sigma)*t) in general (they
# coincide only for band-picking, a pure per-lane selection, since selection
# commutes with exp). Verified 2026-09-10 that the smooth upsampler's grey
# invariant survives exactly (spec_refl_unbounded(c,c,c) -> c in every lane,
# to machine precision) — see project_spectral_media_state.md,
# "Refutation 2" — so nothing is lost by switching sigma_t's upsampling from
# band-picking to the real curve; a grey medium is bit-for-bit unaffected.
#
# This computes the weight the CORRECT way: upsample sigma_t to per-lane
# values FIRST, then exponentiate PER LANE, using the same red-channel
# proposal density `sample_homogeneous_free_flight` already sampled `t_free`
# from (so this is purely a change to the WEIGHT, never to what distance was
# sampled or its acceptance probability -- no change to path continuation).
@always_inline
def medium_sigma_t_spectral(
    med: Medium_C, wavelengths: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
) -> SpectralSample:
    """Upsample a medium's total extinction sigma_t = sigma_a + sigma_s from
    its 3 authored RGB channels to the 4 hero wavelengths via
    spec_refl_unbounded -- the same unbounded-coefficient upsampler used for
    reflectance-shaped weights that may exceed 1 (an extinction coefficient
    is the same kind of quantity). Shared by spectral_free_flight_weight,
    bdpt.mojo's _visible_transmittance and gpu.mojo's _sample_medium_core
    (via medium_transmittance_ratio_spectral below) so all consumers agree on
    what "the medium's colour" means."""
    var sig_t = med.sigma_a + med.sigma_s
    return spec_refl_unbounded(
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        sig_t.r, sig_t.g, sig_t.b, wavelengths)

@always_inline
def medium_transmittance_ratio_spectral(
    med: Medium_C, t: Float32, wavelengths: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
) -> SpectralSample:
    """The chromatic RATIO of each hero wavelength's transmittance over a
    homogeneous segment of length `t` to the sampled (red) channel's:
    exp(-(sigma_t(lambda_i) - sigma_t.r) * t), sigma_t upsampled to the 4
    hero lanes FIRST (medium_sigma_t_spectral) then exponentiated PER LANE --
    exactly the `d0..d3` half of spectral_free_flight_weight below, pulled
    out because gpu.mojo's _sample_medium_core needs precisely this ratio
    (and nothing else -- it applies its own sigma_s/albedo factor separately,
    via a real scatter/absorb coin flip rather than a deterministic
    multiply) and can share this ordering-sensitive arithmetic without also
    taking on BDPT/SPPM's deterministic-continuation collision weight. Never
    fold a sigma_s/albedo factor into THIS function -- see the two callers'
    own docstrings for why they apply it differently."""
    var sig_t_r = med.sigma_a.r + med.sigma_s.r
    if sig_t_r <= Float32(0.0):
        return SpectralSample(Float32(1.0))
    var sig_t_spec = medium_sigma_t_spectral(
        med, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
    return SpectralSample(
        exp(-(sig_t_spec.v0 - sig_t_r) * t),
        exp(-(sig_t_spec.v1 - sig_t_r) * t),
        exp(-(sig_t_spec.v2 - sig_t_r) * t),
        exp(-(sig_t_spec.v3 - sig_t_r) * t))

@always_inline
def spectral_free_flight_weight(
    med: Medium_C, ff: HomogeneousFreeFlight, t_surf: Float32, wavelengths: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
) -> SpectralSample:
    """Replaces `rgb_bands_to_spectral_sample(ff.weight.r, .g, .b, wl)` at
    every chromatic-media consumer. `ff` must come from
    `sample_homogeneous_free_flight(med, t_surf, ...)` -- SAME `med` and
    `t_surf` -- (this does not re-sample anything, it only re-derives the
    weight spectrally). `t_surf` is required explicitly, not read from
    `ff.t_free`: in the pass-through branch `ff.t_free` is the raw sampled
    distance (which exceeded `t_surf`, that's WHY it's pass-through), while
    the weight must use the actual segment length `t_surf` -- the same
    distinction `sample_homogeneous_free_flight`'s own RGB `Tr` makes
    (built from `t_surf`, not `t_free`, in that branch).
    Grey media pass through exactly, at every tau, since spec_refl_unbounded
    reproduces a grey coefficient exactly and the sig_t_r reference cancels
    to 1 on every lane. See docs/02_spectra_and_color.md, "Chromatic
    extinction" for the derivation and the measured before/after."""
    var sig_t_r = med.sigma_a.r + med.sigma_s.r
    if sig_t_r <= Float32(0.0):
        return SpectralSample(Float32(1.0))
    # sig_t_spec computed ONCE and reused for both the d0..d3 ratio and (on
    # collision) the extra r0..r3 factor -- deliberately NOT routed through
    # medium_transmittance_ratio_spectral, which would upsample sigma_t a
    # second time on the collision branch (that helper exists for gpu.mojo's
    # simpler case, which never needs r0..r3).
    var sig_t_spec = medium_sigma_t_spectral(
        med, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
    var t = ff.t_free if ff.collided else t_surf
    var d0 = exp(-(sig_t_spec.v0 - sig_t_r) * t)
    var d1 = exp(-(sig_t_spec.v1 - sig_t_r) * t)
    var d2 = exp(-(sig_t_spec.v2 - sig_t_r) * t)
    var d3 = exp(-(sig_t_spec.v3 - sig_t_r) * t)
    if not ff.collided:
        return SpectralSample(d0, d1, d2, d3)
    # Collision branch carries an extra sigma_t(lambda)/sigma_t.r factor (see
    # sample_homogeneous_free_flight's own derivation comment for the RGB
    # analogue -- same algebra, per hero lane instead of per RGB channel).
    var r0 = sig_t_spec.v0 / sig_t_r
    var r1 = sig_t_spec.v1 / sig_t_r
    var r2 = sig_t_spec.v2 / sig_t_r
    var r3 = sig_t_spec.v3 / sig_t_r
    return SpectralSample(d0 * r0, d1 * r1, d2 * r2, d3 * r3)


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
struct NvdbGrid_C(TrivialRegisterPassable):
    """Sparse heterogeneous density grid backing a Medium_C (PBRT-v4
    "nanovdb" -- a decompressed .nvdb blob, sampled via nvdb_sample_index).
    Sibling to Grid_C, not a variant of it: NanoVDB's own index space has a
    DIFFERENT coordinate convention (integer voxel index, own affine map)
    from Grid_C's dense-array [p0,p1]-box convention, so this is its own
    struct rather than a `kind` flag bolted onto Grid_C.

    Two transforms compose to go from pbrt world space to an nvdb voxel
    index, mirroring the two-stage pipeline pbrt-v4 itself uses for nanovdb
    media: world_to_medium (this medium's own pbrt CTM inverse, same 4x4
    column-major convention as Grid_C's field of the same name) maps pbrt
    world space into the .nvdb file's OWN embedded world space, and
    inv_map/map_vec (that file's PNanoVDB "Map", read once from the blob
    at parse time) maps that into fractional index space:
        world_to_index(x) = inv_map * (x - map_vec)
    (matches pnanovdb_map_apply_inverse exactly; inv_map is row-major 3x3,
    packed into a SIMD16 with 7 unused padding lanes to reuse Grid_C's own
    storage convention rather than invent a 9-wide one).

    index_min/index_max are the blob's indexBBox (nvdb_index_bbox), for a
    cheap reject before touching the blob at all. max_density is the
    majorant (root-node max, nvdb_value_range) used for delta-tracking free
    -flight sampling, same role as Grid_C.max_density -- coarser than a
    per-leaf majorant would be, a documented, deliberate v1 scope choice.
    """
    var blob: UnsafePointer[UInt8, MutExternalOrigin]
    var blob_size: Int64  # bytes -- CPU sampling never needs this (pure offset
                           # arithmetic, no bounds check), only the GPU upload's
                           # memcpy does; kept here rather than threaded as a
                           # separate parallel array alongside every other field.
    var world_to_medium: SIMD[DType.float32, 16]
    var inv_map: SIMD[DType.float32, 16]  # row-major 3x3 in lanes 0..8, rest unused
    var map_vec: Vec3f
    var index_min: Point3f
    var index_max: Point3f
    var max_density: Float32

@always_inline
def nvdb_sample_density(grid: NvdbGrid_C, p_world: Vec3f) -> Float32:
    """Point-sampled (NOT trilinear -- v1 scope, see project_nanovdb_media
    memory) density at a world-space point; 0 outside the grid's index
    bounds. Same "0 outside bounds" contract as grid_sample_density, so a
    shadow ray's ratio-tracking transmittance naturally stops attenuating
    once it exits the medium with no extra bookkeeping, exactly as that
    function's own docstring notes."""
    var m = grid.world_to_medium
    var mx = m[0]*p_world[0] + m[4]*p_world[1] + m[8]*p_world[2] + m[12]
    var my = m[1]*p_world[0] + m[5]*p_world[1] + m[9]*p_world[2] + m[13]
    var mz = m[2]*p_world[0] + m[6]*p_world[1] + m[10]*p_world[2] + m[14]

    var im = grid.inv_map
    var sx = mx - grid.map_vec.x
    var sy = my - grid.map_vec.y
    var sz = mz - grid.map_vec.z
    var ix = sx*im[0] + sy*im[1] + sz*im[2]
    var iy = sx*im[3] + sy*im[4] + sz*im[5]
    var iz = sx*im[6] + sy*im[7] + sz*im[8]

    # Trilinear, matching pbrt's NanoVDBMedium, which samples density with
    # nanovdb's SampleFromVoxels<..., 1, false> (order 1 = trilinear). NanoVDB's
    # convention places voxel VALUES at integer index coordinates, so the base
    # cell is floor(p) and the weights are the fractional part -- this is NOT
    # the half-integer cell-centre convention Grid_C's dense sampler uses, and
    # getting that wrong shifts the field by half a voxel.
    #
    # Point sampling (the original v1 scope) is not merely noisier: on a sparse
    # wispy cloud it keeps hard voxel edges where pbrt's trilinear bleeds each
    # occupied voxel into its neighbours, which changes the effective optical
    # thickness and rendered a visibly dimmer cloud than the reference.
    var fi = floor(ix); var fj = floor(iy); var fk = floor(iz)
    var i = Int32(fi); var j = Int32(fj); var k = Int32(fk)
    # Reject only when the whole 8-tap neighbourhood lies outside the grid;
    # taps that fall outside individually just read the background (0).
    if i < Int32(grid.index_min.x) - Int32(1) or i > Int32(grid.index_max.x): return Float32(0.0)
    if j < Int32(grid.index_min.y) - Int32(1) or j > Int32(grid.index_max.y): return Float32(0.0)
    if k < Int32(grid.index_min.z) - Int32(1) or k > Int32(grid.index_max.z): return Float32(0.0)
    var tx = ix - fi; var ty = iy - fj; var tz = iz - fk
    var v000: Float32; var v100: Float32; var v010: Float32; var v110: Float32
    var v001: Float32; var v101: Float32; var v011: Float32; var v111: Float32
    # Fast path: when the 2x2x2 stencil does not cross a leaf boundary (the
    # low 3 bits of every axis are < 7), one root->leaf descent locates the
    # leaf and the other seven values are pure offset arithmetic inside it --
    # 1 tree walk instead of 8. A leaf is 8^3, so this covers (7/8)^3 ~ 67%
    # of lookups; the rest fall back to eight independent descents, which is
    # exactly what this function did before. Both paths return identical
    # values by construction (the slow path would descend to this same leaf).
    var leaf_base = -1
    if (i & Int32(7)) < Int32(7) and (j & Int32(7)) < Int32(7) and (k & Int32(7)) < Int32(7):
        leaf_base = nvdb_leaf_base(grid.blob, i, j, k)
    if leaf_base >= 0:
        v000 = nvdb_leaf_value(grid.blob, leaf_base, i,           j,           k)
        v100 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j,           k)
        v010 = nvdb_leaf_value(grid.blob, leaf_base, i,           j + Int32(1), k)
        v110 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j + Int32(1), k)
        v001 = nvdb_leaf_value(grid.blob, leaf_base, i,           j,           k + Int32(1))
        v101 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j,           k + Int32(1))
        v011 = nvdb_leaf_value(grid.blob, leaf_base, i,           j + Int32(1), k + Int32(1))
        v111 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j + Int32(1), k + Int32(1))
    else:
        v000 = nvdb_sample_index(grid.blob, i,           j,           k)
        v100 = nvdb_sample_index(grid.blob, i + Int32(1), j,           k)
        v010 = nvdb_sample_index(grid.blob, i,           j + Int32(1), k)
        v110 = nvdb_sample_index(grid.blob, i + Int32(1), j + Int32(1), k)
        v001 = nvdb_sample_index(grid.blob, i,           j,           k + Int32(1))
        v101 = nvdb_sample_index(grid.blob, i + Int32(1), j,           k + Int32(1))
        v011 = nvdb_sample_index(grid.blob, i,           j + Int32(1), k + Int32(1))
        v111 = nvdb_sample_index(grid.blob, i + Int32(1), j + Int32(1), k + Int32(1))
    var v00 = v000 + (v100 - v000) * tx
    var v10 = v010 + (v110 - v010) * tx
    var v01 = v001 + (v101 - v001) * tx
    var v11 = v011 + (v111 - v011) * tx
    var v0 = v00 + (v10 - v00) * ty
    var v1 = v01 + (v11 - v01) * ty
    return v0 + (v1 - v0) * tz

@always_inline
def _slab_range(o: Vec3f, d: Vec3f, bmin: Vec3f, bmax: Vec3f) -> SIMD[DType.float32, 2]:
    """Ray/AABB slab test returning (t_enter, t_exit); t_exit < t_enter means
    "misses the box". The ray is NOT normalized here on purpose: `d` is passed
    in the SAME space as the box, transformed by the same affine map as `o`,
    so the returned t values stay in the caller's original (world) parameter
    units and can be compared directly against a world-space distance."""
    var t0 = Float32(-1.0e30)
    var t1 = Float32(1.0e30)
    for a in range(3):
        var di = d[a]
        var oi = o[a]
        var lo = bmin[a]
        var hi = bmax[a]
        if abs(di) < Float32(1e-12):
            if oi < lo or oi > hi:
                return SIMD[DType.float32, 2](Float32(1.0), Float32(-1.0))  # miss
        else:
            var inv = Float32(1.0) / di
            var ta = (lo - oi) * inv
            var tb = (hi - oi) * inv
            var tmin_a = ta if ta < tb else tb
            var tmax_a = tb if ta < tb else ta
            if tmin_a > t0: t0 = tmin_a
            if tmax_a < t1: t1 = tmax_a
    return SIMD[DType.float32, 2](t0, t1)

@always_inline
def nvdb_ray_range(grid: NvdbGrid_C, org: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 2]:
    """(t_enter, t_exit) of the ray against this grid's index bbox, in WORLD t
    units. Needed because an infinite (environment) light's shadow ray has no
    finite distance to march: ratio-tracking it at the majorant step rate to a
    nominal "infinity" would burn millions of iterations through empty space
    for a transmittance that stops changing the moment the ray leaves the grid.
    Both transforms are affine, so t is preserved and the range can be
    intersected directly with the world-space shadow-ray length."""
    var m = grid.world_to_medium
    var ox = m[0]*org[0] + m[4]*org[1] + m[8]*org[2] + m[12]
    var oy = m[1]*org[0] + m[5]*org[1] + m[9]*org[2] + m[13]
    var oz = m[2]*org[0] + m[6]*org[1] + m[10]*org[2] + m[14]
    var dx = m[0]*dir[0] + m[4]*dir[1] + m[8]*dir[2]
    var dy = m[1]*dir[0] + m[5]*dir[1] + m[9]*dir[2]
    var dz = m[2]*dir[0] + m[6]*dir[1] + m[10]*dir[2]
    var im = grid.inv_map
    var sx = ox - grid.map_vec.x
    var sy = oy - grid.map_vec.y
    var sz = oz - grid.map_vec.z
    var oi = Vec3f(sx*im[0] + sy*im[1] + sz*im[2],
                   sx*im[3] + sy*im[4] + sz*im[5],
                   sx*im[6] + sy*im[7] + sz*im[8])
    var di = Vec3f(dx*im[0] + dy*im[1] + dz*im[2],
                   dx*im[3] + dy*im[4] + dz*im[5],
                   dx*im[6] + dy*im[7] + dz*im[8])
    return _slab_range(oi, di,
        Vec3f(grid.index_min.x, grid.index_min.y, grid.index_min.z),
        Vec3f(grid.index_max.x + Float32(1), grid.index_max.y + Float32(1), grid.index_max.z + Float32(1)))

@always_inline
def nvdb_index_ray(grid: NvdbGrid_C, org: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 8]:
    """The world ray expressed in the grid's INDEX space, as
    (ox,oy,oz,|d|, dx,dy,dz,unused). Both transforms are affine and the
    direction is carried as a vector, so the ray parameter t is IDENTICAL in
    both spaces -- index-space box tests therefore return world-space t
    directly. Computed once per tracked ray so the per-segment majorant
    walk costs only arithmetic."""
    var m = grid.world_to_medium
    var ox = m[0]*org[0] + m[4]*org[1] + m[8]*org[2] + m[12]
    var oy = m[1]*org[0] + m[5]*org[1] + m[9]*org[2] + m[13]
    var oz = m[2]*org[0] + m[6]*org[1] + m[10]*org[2] + m[14]
    var dx = m[0]*dir[0] + m[4]*dir[1] + m[8]*dir[2]
    var dy = m[1]*dir[0] + m[5]*dir[1] + m[9]*dir[2]
    var dz = m[2]*dir[0] + m[6]*dir[1] + m[10]*dir[2]
    var im = grid.inv_map
    var sx = ox - grid.map_vec.x
    var sy = oy - grid.map_vec.y
    var sz = oz - grid.map_vec.z
    var iox = sx*im[0] + sy*im[1] + sz*im[2]
    var ioy = sx*im[3] + sy*im[4] + sz*im[5]
    var ioz = sx*im[6] + sy*im[7] + sz*im[8]
    var idx = dx*im[0] + dy*im[1] + dz*im[2]
    var idy = dx*im[3] + dy*im[4] + dz*im[5]
    var idz = dx*im[6] + dy*im[7] + dz*im[8]
    var dlen = sqrt(max(idx*idx + idy*idy + idz*idz, Float32(1e-20)))
    return SIMD[DType.float32, 8](iox, ioy, ioz, dlen, idx, idy, idz, Float32(0))

@always_inline
def nvdb_node_exit_t(iray: SIMD[DType.float32, 8], t_now: Float32, dim: Float32) -> Float32:
    """t at which the ray leaves the `dim`-aligned node box currently
    containing it. `dim` is a power of two (8/128/4096, the leaf/lower/upper
    extents), so the box base is the index coordinate with its low bits
    masked off -- an arithmetic shift, which is correct for negative
    coordinates too. A small nudge past the face is added so the next
    majorant query lands in the NEXT node and the walk always makes
    progress; without it a ray exactly on a node face would re-query the
    same node forever."""
    var px = iray[0] + t_now * iray[4]
    var py = iray[1] + t_now * iray[5]
    var pz = iray[2] + t_now * iray[6]
    var shift = Int32(3)
    if dim > Float32(2048): shift = Int32(12)
    elif dim > Float32(64): shift = Int32(7)
    var b0 = Float32((Int32(floor(px)) >> shift) << shift)
    var b1 = Float32((Int32(floor(py)) >> shift) << shift)
    var b2 = Float32((Int32(floor(pz)) >> shift) << shift)
    var t_exit = Float32(1.0e30)
    if abs(iray[4]) > Float32(1e-12):
        var bx = (b0 + dim) if iray[4] > Float32(0) else b0
        var tx = (bx - iray[0]) / iray[4]
        if tx < t_exit: t_exit = tx
    if abs(iray[5]) > Float32(1e-12):
        var by = (b1 + dim) if iray[5] > Float32(0) else b1
        var ty = (by - iray[1]) / iray[5]
        if ty < t_exit: t_exit = ty
    if abs(iray[6]) > Float32(1e-12):
        var bz = (b2 + dim) if iray[6] > Float32(0) else b2
        var tz = (bz - iray[2]) / iray[6]
        if tz < t_exit: t_exit = tz
    # nudge ~1e-3 voxel past the face
    var eps = Float32(1.0e-3) / iray[3]
    if t_exit <= t_now: t_exit = t_now
    return t_exit + eps

@always_inline
def nvdb_majorant_at_world(grid: NvdbGrid_C, p_world: Vec3f) -> SIMD[DType.float32, 2]:
    """LOCAL majorant (max, extent) at a WORLD-space point -- thin wrapper
    over nanovdb.mojo's nvdb_majorant_at that applies the same two-stage
    world->medium->index transform nvdb_sample_density uses."""
    var m = grid.world_to_medium
    var mx = m[0]*p_world[0] + m[4]*p_world[1] + m[8]*p_world[2] + m[12]
    var my = m[1]*p_world[0] + m[5]*p_world[1] + m[9]*p_world[2] + m[13]
    var mz = m[2]*p_world[0] + m[6]*p_world[1] + m[10]*p_world[2] + m[14]
    var im = grid.inv_map
    var sx = mx - grid.map_vec.x
    var sy = my - grid.map_vec.y
    var sz = mz - grid.map_vec.z
    var ix = sx*im[0] + sy*im[1] + sz*im[2]
    var iy = sx*im[3] + sy*im[4] + sz*im[5]
    var iz = sx*im[6] + sy*im[7] + sz*im[8]
    return nvdb_majorant_at(grid.blob, Int32(floor(ix)), Int32(floor(iy)), Int32(floor(iz)))

@always_inline
def grid_ray_range(grid: Grid_C, org: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 2]:
    """(t_enter, t_exit) of the ray against a dense grid's [p0,p1] box, in
    WORLD t units. Same purpose as nvdb_ray_range -- see that docstring."""
    var m = grid.world_to_medium
    var o = Vec3f(m[0]*org[0] + m[4]*org[1] + m[8]*org[2] + m[12],
                  m[1]*org[0] + m[5]*org[1] + m[9]*org[2] + m[13],
                  m[2]*org[0] + m[6]*org[1] + m[10]*org[2] + m[14])
    var d = Vec3f(m[0]*dir[0] + m[4]*dir[1] + m[8]*dir[2],
                  m[1]*dir[0] + m[5]*dir[1] + m[9]*dir[2],
                  m[2]*dir[0] + m[6]*dir[1] + m[10]*dir[2])
    return _slab_range(o, d,
        Vec3f(grid.p0.x, grid.p0.y, grid.p0.z), Vec3f(grid.p1.x, grid.p1.y, grid.p1.z))

@always_inline
@always_inline
def _bb_lobe(x: Float32, m: Float32, s1: Float32, s2: Float32) -> Float32:
    """One asymmetric-Gaussian lobe of Wyman et al. 2013's analytic fit to the
    CIE 1931 colour-matching functions."""
    var sd = s1 if x < m else s2
    var t = (x - m) / sd
    return exp(Float32(-0.5) * t * t)


def blackbody_rgb(temp: Float32) -> RGB:
    """Linear-sRGB colour of a `temp` Kelvin blackbody, normalized to
    LUMINANCE 1 -- which is what pbrt's blackbody lights actually deliver, so
    a `"float scale"` beside a `"blackbody L"` means the same thing in both
    renderers.

    This used to be the Mitchell-Charity RGB approximation normalized so the
    max CHANNEL was 1, on the stated reasoning that pbrt's BlackbodySpectrum
    "likewise normalizes to a peak of 1 (via Wien's law)". That conflates two
    different things: pbrt normalizes the SPECTRUM's peak, and the RGB that
    spectrum then integrates to has neither unit max-channel nor unit
    luminance a priori. Measured against pbrt on a quad light (per unit L,
    with an `"rgb L" [1 1 1]` control confirming the harness at 1.0006):

        T=3500  pbrt [1.5642 0.8913 0.4063]   old code [1.0 0.756 0.555]
        T=6500  pbrt [1.0437 0.9827 1.0335]   old code [1.0 1.0   0.985]

    -- wrong in magnitude AND chromaticity. pbrt's own values have luminance
    0.9994 and 0.9994 respectively, i.e. exactly 1, which is what this now
    reproduces (to 0.7% at 3500 K and 0.2% at 6500 K, the residual being the
    analytic CIE fit rather than the quadrature).

    Planck's law integrated against Wyman et al. 2013's analytic CIE fits at
    10 nm over 360-830 nm, then XYZ->linear sRGB and divided by Y. 10 nm is
    indistinguishable from 1 nm here (<0.01%) because Planck is smooth and the
    fit's narrowest lobe is still ~12 nm wide -- worth keeping cheap, since
    this also runs per emission sample inside the GPU medium kernel for nanovdb
    temperature grids. Wavelengths are carried in MICROMETRES so lambda^-5 stays
    in a comfortable Float32 range.

    A future improvement worth noting: for the SPECTRAL medium path this is
    strictly worse than evaluating Planck at the four hero wavelengths
    directly -- that would be both cheaper (4 exps, not 48) and exact, with no
    RGB round trip at all."""
    if temp <= Float32(0.0):
        return RGB(Float32(0.0))
    var X = Float32(0.0)
    var Y = Float32(0.0)
    var Z = Float32(0.0)
    var lam = Float32(360.0)
    while lam <= Float32(830.0):
        var lum = lam * Float32(0.001)                  # micrometres
        var xarg = Float32(14388.0) / (lum * temp)      # c2 in um*K
        var inv: Float32
        if xarg > Float32(80.0):
            # exp(xarg) would overflow Float32; 1/(e^x - 1) -> e^-x there.
            inv = exp(-xarg)
        else:
            inv = Float32(1.0) / (exp(xarg) - Float32(1.0))
        var l2 = lum * lum
        var spec = inv / (l2 * l2 * lum)
        X += spec * (Float32(1.056) * _bb_lobe(lam, Float32(599.8), Float32(37.9), Float32(31.0))
                   + Float32(0.362) * _bb_lobe(lam, Float32(442.0), Float32(16.0), Float32(26.7))
                   - Float32(0.065) * _bb_lobe(lam, Float32(501.1), Float32(20.4), Float32(26.2)))
        Y += spec * (Float32(0.821) * _bb_lobe(lam, Float32(568.8), Float32(46.9), Float32(40.5))
                   + Float32(0.286) * _bb_lobe(lam, Float32(530.9), Float32(16.3), Float32(31.1)))
        Z += spec * (Float32(1.217) * _bb_lobe(lam, Float32(437.0), Float32(11.8), Float32(36.0))
                   + Float32(0.681) * _bb_lobe(lam, Float32(459.0), Float32(26.0), Float32(13.8)))
        lam += Float32(10.0)
    if Y <= Float32(0.0):
        return RGB(Float32(0.0))
    var iy = Float32(1.0) / Y
    var xn = X * iy
    var zn = Z * iy
    var r = Float32(3.2406) * xn - Float32(1.5372) - Float32(0.4986) * zn
    var g = Float32(-0.9689) * xn + Float32(1.8758) + Float32(0.0415) * zn
    var b = Float32(0.0557) * xn - Float32(0.2040) + Float32(1.0570) * zn
    return RGB(max(r, Float32(0.0)), max(g, Float32(0.0)), max(b, Float32(0.0)))


@always_inline
def hg_phase(cos_theta: Float32, g: Float32) -> Float32:
    """Henyey-Greenstein phase function, pbrt's exact form and convention:
    with `cos_theta = dot(wo, wi)` and wo pointing BACK along the incoming ray,
    g > 0 peaks at cos_theta = -1, i.e. wi continuing forward. Integrates to 1
    over the sphere, so it doubles as its own sampling pdf."""
    var gg = g * g
    var denom = Float32(1.0) + gg + Float32(2.0) * g * cos_theta
    if denom < Float32(1e-7):
        denom = Float32(1e-7)
    return INV_FOUR_PI * (Float32(1.0) - gg) / (denom * sqrt(denom))

@always_inline
def hg_sample(wo: Vec3f, g: Float32, u1: Float32, u2: Float32) -> SIMD[DType.float32, 4]:
    """Sample the HG phase function about `wo`; returns (wi.x, wi.y, wi.z, pdf).
    Falls back to the uniform-sphere sampler for |g| < 1e-3, both because the
    closed form is numerically unstable there and because that IS the isotropic
    case. Same inversion pbrt uses."""
    var cos_theta: Float32
    if abs(g) < Float32(1e-3):
        cos_theta = Float32(1.0) - Float32(2.0) * u1
    else:
        var gg = g * g
        var sq = (Float32(1.0) - gg) / (Float32(1.0) + g - Float32(2.0) * g * u1)
        cos_theta = -(Float32(1.0) + gg - sq * sq) / (Float32(2.0) * g)
    if cos_theta < Float32(-1.0): cos_theta = Float32(-1.0)
    if cos_theta > Float32(1.0): cos_theta = Float32(1.0)
    var sin_theta = sqrt(max(Float32(0.0), Float32(1.0) - cos_theta * cos_theta))
    var phi = TWO_PI * u2
    var f = Frame.from_z(wo)
    var lx = sin_theta * cos(phi)
    var ly = sin_theta * sin(phi)
    var wi = f.x * lx + f.y * ly + f.z * cos_theta
    return SIMD[DType.float32, 4](wi[0], wi[1], wi[2], hg_phase(cos_theta, g))

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
    var cdf_ptr: UnsafePointer[Float32, MutExternalOrigin]   # flat 2D CDF (marginal + conditional)
    var pixels_ptr: UnsafePointer[Float32, MutExternalOrigin] # raw HDR pixels, 3 floats/pixel (CPU only)
    var world_to_light: UnsafePointer[Float32, MutExternalOrigin]  # 16-float col-major inverse of light CTM

# ── GPU / render pipeline helpers ─────────────────────────────────────────────

@fieldwise_init
struct GpuTexture_C(TrivialRegisterPassable):
    comptime FORMAT_F32 = 0   # data holds Float32 linear RGB
    comptime FORMAT_U8 = 1    # data holds UInt8, decoded to linear through lut
    var data: UnsafePointer[UInt8, MutExternalOrigin]    # device pointer: full mip pyramid, contiguous
    var lut: UnsafePointer[Float32, MutExternalOrigin]   # device pointer: 256-entry byte -> linear table (FORMAT_U8 only)
    var width: Int32                                     # level-0 width
    var height: Int32                                    # level-0 height
    var n_levels: Int32                                  # number of mip levels stored in `data` (>=1)
    var channels: Int32                                  # stored channels per texel: 1 (replicated to RGB) or 3
    var format: Int32                                    # FORMAT_F32 or FORMAT_U8

@fieldwise_init
struct NormalSlopeMap_C(TrivialRegisterPassable):
    """A normal map in LEAN SLOPE space, level 0 only -- the representation
    the reference SMS renderer's manifold walk reads (render/normalmap.h
    `eval_normal`/`eval_normal_derivatives` with `use_slopes=true`).

    Ordinary shading samples a normal map through `sample_texture`, which is
    enough when only the normal itself is needed. A manifold walk needs the
    normal's DERIVATIVES too (they are what the specular constraint's
    Jacobian is built from), and bilinear interpolation of the raw RGB gives
    a derivative that disagrees with the reference's. So the map is stored a
    second time, converted once at scene-build time:

        n = normalize(2*rgb - 1);  slope = (-n.x/n.z, -n.y/n.z)

    with the local normal recovered as the (unnormalized) `(-sx, -sy, 1)`.
    Interpolating in slope space is what makes the derivative analytic and
    exactly matches the reference; it is also invariant to the RGB
    normalization, so both spellings agree texel-for-texel and differ only
    BETWEEN texels.

    Square, power-of-two maps only (the reference assumes the same).
    `res <= 0` means "this texture has no slope map" -- always test that
    before touching `slopes`, which is dangling in that case."""
    var slopes: UnsafePointer[Float32, MutExternalOrigin]  # 2 floats/texel, row-major
    var res:    Int32                                      # 0 => absent

@always_inline
def normal_slope_map_none() -> NormalSlopeMap_C:
    return NormalSlopeMap_C(UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(), Int32(0))

@fieldwise_init
struct ShadowTask_C(TrivialRegisterPassable):
    """A deferred shadow ray with its pre-computed radiance contribution."""
    var origin: Point3f
    var direction: Vec3f
    var tmax: Float32
    var contrib: SpectralSample   # deferred TRANSPORT, so spectral like throughput
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
    var estimate: RGB
    var albedo: RGB
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

# pbrt's coateddiffuse/coatedconductor default when a scene doesn't set
# "float thickness" explicitly (LayeredBxDF's own default). gonzales has no
# per-material storage for this (Material_C has no thickness field -- adding
# one touches the GPU upload path/struct size, out of scope here), so every
# coat uses this single default. Only 2 of the pbrt-v4 corpus's ~180
# coateddiffuse/coatedconductor materials (both bistro_cafe coatedconductor
# props, not a dominant surface) set it explicitly; this default covers the
# rest exactly.
comptime DEFAULT_COAT_THICKNESS: Float32 = 0.01

@always_inline
def coat_beer_lambert_tr(cos_theta_internal: Float32, thickness: Float32) -> Float32:
    """Beer-Lambert transmittance for one crossing of a coat layer of the
    given thickness, given the INTERNAL (refracted) cosine of the ray's angle
    to the interface normal. Mirrors pbrt's LayeredBxDF::Tr(dz, w) =
    exp(-|dz / w.z|) -- see bxdfs.h. `cos_theta_internal` is w.z in that
    formula; callers crossing from outside the coat must refract the external
    cosine first via cos_theta_t_dielectric below (pbrt tracks z in the
    medium's own frame, not the external ray's)."""
    var c = max(cos_theta_internal, Float32(1e-4))
    return exp(-thickness / c)

@always_inline
def cos_theta_t_dielectric(cos_theta_i_in: Float32, eta: Float32) -> Float32:
    """Cosine of the refracted angle inside a medium of relative IOR eta
    (eta_t/eta_i), given the EXTERNAL cosine of incidence. Same geometry as
    fr_dielectric, factored out so coat-thickness attenuation (which needs
    the internal ray angle, not the external one) can share it. Returns 0 on
    total internal reflection (grazing limit, matching fr_dielectric's own
    TIR branch)."""
    var cos_theta_i = max(Float32(0.0), min(Float32(1.0), abs(cos_theta_i_in)))
    var sin2_theta_i = max(Float32(0.0), Float32(1.0) - cos_theta_i * cos_theta_i)
    var sin2_theta_t = sin2_theta_i / max(eta * eta, Float32(1e-6))
    if sin2_theta_t >= Float32(1.0):
        return Float32(0.0)
    return safe_sqrt(Float32(1.0) - sin2_theta_t)

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
