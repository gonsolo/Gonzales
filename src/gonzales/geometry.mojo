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
comptime INV_TWO_PI : Float32 = 0.15915494309189533577
comptime INV_FOUR_PI: Float32 = 0.07957747154594766788

# PathState_C.lastBsdfPdf sentinel: "a real scatter happened here, but its
# direct-light term was already reported by NEE, so every emitter/miss
# handler must contribute ZERO for it and let the ray carry indirect light
# only." Used by the layered coat exit, whose true pdf is intractable to
# MIS-combine. It has to be distinguishable from 0.0, because a CAMERA ray
# also carries lastBsdfPdf == 0 and must take the full background instead.
# The lastBsdfPdf / last_bsdf_pdf SENTINEL SPACE -- every negative value, in
# ONE place, because they collided twice. A real pdf is >= 0; each negative
# value below is a distinct instruction to the emitter/miss handlers, and the
# two integrators must agree on all of them.
#
#   PDF_DELTA_FULL     bdpt: a delta bounce, no NEE happened at that vertex,
#                      so an emitter hit takes FULL weight.
#   PDF_VOL_PHASE_HIT  bdpt: a volume phase scatter hit an emitter; weight
#                      against the phase pdf (INV_FOUR_PI), not a BSDF pdf.
#   PDF_DROP_DIRECT    both: NEE already reported this ray's DIRECT term (the
#                      layered coat's exit ray); contribute indirect only.
#
# History, so the next value is chosen by looking here rather than guessing:
# PDF_DROP_DIRECT was first -1 (collided with PDF_DELTA_FULL: opposite
# meanings), then -2 (collided with PDF_VOL_PHASE_HIT: at bdpt's emitter gates
# the drop branch shadowed the phase branch, leaving its default weight of 1
# -- a double count that read +93% on a medium lit by an emissive sphere).
comptime PDF_DELTA_FULL:    Float32 = -1.0
comptime PDF_VOL_PHASE_HIT: Float32 = -2.0
comptime PDF_DROP_DIRECT:   Float32 = -3.0

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

struct LobeKind:
    """The BSDF a STORED vertex is re-evaluated with when something connects
    to it later (VCM's BDPTVertex, SPPM's visible point, the BxDF NEE helpers).
    Not MatKind: several materials share one lobe, and a coated or subsurface
    material stores a lobe that is not the material's own. One numbering for
    every integrator, so a new kind cannot silently collide with an existing
    one (kind 4 once meant coateddiffuse in VCM and BSSRDF in SPPM)."""
    comptime lambertian  = Int32(0)
    comptime ggx         = Int32(1)
    comptime hair        = Int32(2)
    comptime measured    = Int32(3)
    comptime coated_walk = Int32(4)   # coateddiffuse walk EXIT: base seen through the coat
    comptime bssrdf      = Int32(5)   # subsurface: exit lobe Ft(cos)/pi (VCM), diffusion gather (SPPM)
    comptime diffuse_transmit = Int32(6)   # two cosine lobes, one per side
    # The coat's OWN glossy reflection off the top interface -- a different
    # physical event from coated_walk, which is the base seen THROUGH the
    # coat. Both outcomes of the same walk, so both were stored as
    # coated_walk until this existed, and lobe_eval then evaluated the
    # glossy bounce with coat_eval_smooth -- the base-transmission model,
    # complete with the base's albedo, for a reflection that never reaches
    # the base at all. Splitting them is what lets each have its own
    # evaluator; see lobe_eval's two branches.
    comptime coated_reflect = Int32(7)
    comptime layered = Int32(8)   # coateddiffuse as pbrt's LayeredBxDF (layered.mojo): f, pdf_fwd, pdf_rev all real

struct PhotonKind:
    """What an SPPM photon (or visible point) was deposited on. A gather only
    pairs equal kinds."""
    comptime surface = Int32(0)
    comptime volume  = Int32(1)
    comptime bssrdf  = Int32(2)   # photons on a subsurface boundary, for the diffusion gather

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
    # For a subsurface boundary (sss_boundary == 1): the MEAN reflectance the
    # interior medium's coefficients were inverted from. The interior is one
    # homogeneous medium, so a textured `reflectance` has to be collapsed to a
    # single value to build it -- but the diffuse reflectance a point SHOULD
    # show is its own texel, not that mean, and with the mean alone the head
    # renders flat: measured 23% less spatial detail and 36% less red-channel
    # hue variation than pbrt. `_tex_lookup` already resolves this material's
    # reflectance texture graph, so dividing that lookup by this mean gives
    # the per-point correction. RGB(1) (i.e. inert) for every other material.
    var sss_mean_refl: RGB

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
    var theta_i:     Pointer[Float32, MutUntrackedOrigin]  # [n_theta_i]
    var phi_i:       Pointer[Float32, MutUntrackedOrigin]  # [n_phi_i]
    var wavelengths: Pointer[Float32, MutUntrackedOrigin]  # [n_wavelengths]

    # ndf / sigma: PiecewiseLinear2D<0> — Evaluate-only, no param axes, no CDF.
    var ndf_data:   Pointer[Float32, MutUntrackedOrigin]  # [ndf_ys * ndf_xs]
    var ndf_xs:     Int32
    var ndf_ys:     Int32
    var sigma_data: Pointer[Float32, MutUntrackedOrigin]  # [sigma_ys * sigma_xs]
    var sigma_xs:   Int32
    var sigma_ys:   Int32

    # vndf / luminance: PiecewiseLinear2D<2>, param axes (phi_i, theta_i),
    # both Sample+Evaluate-capable (marginal/conditional CDFs built). Both
    # share the same param resolution, hence the same stride2_* pair below.
    var vndf_data: Pointer[Float32, MutUntrackedOrigin]  # [slices2 * vndf_ys * vndf_xs]
    var vndf_marg: Pointer[Float32, MutUntrackedOrigin]  # [slices2 * vndf_ys]
    var vndf_cond: Pointer[Float32, MutUntrackedOrigin]  # [slices2 * vndf_ys * vndf_xs]
    var vndf_xs:   Int32
    var vndf_ys:   Int32
    var lum_data: Pointer[Float32, MutUntrackedOrigin]   # [slices2 * lum_ys * lum_xs]
    var lum_marg: Pointer[Float32, MutUntrackedOrigin]   # [slices2 * lum_ys]
    var lum_cond: Pointer[Float32, MutUntrackedOrigin]   # [slices2 * lum_ys * lum_xs]
    var lum_xs:   Int32
    var lum_ys:   Int32
    var stride2_phi:   Int32  # PiecewiseLinear2D<2>'s per-param-axis stride
    var stride2_theta: Int32  # (in units of one x*y slice); shared by vndf+luminance

    # spectra: PiecewiseLinear2D<3>, param axes (phi_i, theta_i, wavelengths),
    # Evaluate-only (no CDF) — its own stride triple (different Dimension
    # than vndf/luminance's, so the strides differ even though phi_i/theta_i
    # are shared).
    var spectra_data: Pointer[Float32, MutUntrackedOrigin]  # [slices3 * spectra_ys * spectra_xs]
    var spectra_xs: Int32
    var spectra_ys: Int32
    var stride3_phi:    Int32
    var stride3_theta:  Int32
    var stride3_lambda: Int32

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

