"""Materials, split out of geometry.mojo (the per-cluster module split; see
project_geometry_module_split memory). MatKind/LobeKind/PhotonKind/
Material_C/MeasuredBRDF_C, and schlick_fresnel/fr_dielectric/
DEFAULT_COAT_THICKNESS/coat_beer_lambert_tr/cos_theta_t_dielectric, were two
separate ranges in geometry.mojo; both depend only on the core
(RGB/safe_sqrt), confirmed by a symbol-reference scan of each block with
comments and docstrings stripped before moving it. This is the last of the
six clusters -- geometry.mojo itself keeps only the core (constants,
Point3f/Vec3f/RGB/Frame, and the shared geometry/SIMD helpers)."""
from std.math import min, max, abs, exp
from .geometry import RGB, safe_sqrt

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
# gpu.mojo's scene upload, Stage 3) — mirrors TriangleMesh/Curve_C's
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

