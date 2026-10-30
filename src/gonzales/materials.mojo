"""Material kinds, lobe kinds, the Material record and dielectric Fresnel."""
from std.math import min, max, sqrt
from .geometry import RGB, safe_sqrt, fast_div, fast_recip
from .rng import PCG32

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
    comptime lamp              = Int8(13)   # Cycles' lamp: an area light rays pass after taking its light; no shadows

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
    comptime sss_exit    = Int32(5)   # a subsurface walk's exit, Lambertian; does not merge
    comptime diffuse_transmit = Int32(6)   # two cosine lobes, one per side
    comptime layered = Int32(8)   # coateddiffuse as pbrt's LayeredBxDF (layered.mojo): f, pdf_fwd, pdf_rev all real
    comptime rough_dielectric = Int32(9)   # pbrt DielectricBxDF over Trowbridge-Reitz (layered.mojo diel_*); param = alpha, IOR from the material

    @staticmethod
    @always_inline
    def is_diffuse(k: Int32) -> Bool:
        """A cosine lobe on one or both sides: what MNEE treats as a receiver."""
        return k == LobeKind.lambertian or k == LobeKind.diffuse_transmit

struct PhotonKind:
    """What an SPPM photon (or visible point) was deposited on. A gather only
    pairs equal kinds."""
    comptime surface = Int32(0)
    comptime volume  = Int32(1)

@fieldwise_init
struct Material(TrivialRegisterPassable):
    var type: Int8
    # 1 when this material is the BOUNDARY of a subsurface interior (its
    # medium interface's inside medium has Medium.is_sss). Set by
    # pbrt_parser's medium-interface binding pass. Boundary interactions on
    # such a surface -- entry, exit, and especially total internal reflection
    # -- must NOT be charged to the scene's maxdepth: the whole
    # enter/walk/exit sequence models ONE BSSRDF scattering event, exactly
    # the reasoning that already exempts the interior walk steps via
    # Medium.is_sss. Charging them silently ate light trapped by TIR (a
    # white furnace lost 10.7% at eta 1.5 and 30% at eta 2.0); see
    # Scenes/sss_furnace_sweep.py. Occupies a former padding byte, so
    # Material's size and every existing constructor call site are
    # unchanged.
    var sss_boundary: Int8
    # diffusetransmission only: which lobes the tex_idx texture drives -- 0 both, 1 reflectance only,
    # 2 transmittance only (the other lobe keeps its constant). Occupies a former padding byte.
    var dt_tex_lobes: Int8
    # 1 = a Cycles-style random walk: part of a coateddiffuse base, or a subsurface dielectric's interior (sss_walk_*
    # below). A former padding byte.
    var sss_walk: Int8
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
                              # SceneView.measuredBrdfs (see MeasuredBRDF)
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
    # -1, or a texture of emitted radiance (Blender's Principled emission), added on every hit; not a light to sample.
    var emit_tex_idx: Int32

    # ── What integrators may ask a material; none of them compares `type` itself ──

    @always_inline
    def lobe(self) -> Int32:
        """The LobeKind the BSDF interface evaluates and samples for this material. Smooth glass is a delta chain,
        not a lobe (is_specular_glass)."""
        if self.type == MatKind.diffuse_transmit:
            return LobeKind.diffuse_transmit
        if self.type == MatKind.coated_diffuse or self.type == MatKind.coated_conductor:
            return LobeKind.layered
        if self.type == MatKind.conductor:
            return LobeKind.ggx
        if self.type == MatKind.measured:
            return LobeKind.measured
        if self.type == MatKind.hair:
            return LobeKind.hair
        if self.type == MatKind.dielectric:
            return LobeKind.rough_dielectric
        return LobeKind.lambertian

    @always_inline
    def uses_lobe(self) -> Bool:
        """Scatters through its BSDF (lobe): everything but smooth glass, subsurface boundaries, mixes, emitters
        and pass-through boundaries."""
        if self.type == MatKind.dielectric:
            return self.is_rough_dielectric() and self.sss_boundary == Int8(0)
        return self.type == MatKind.diffuse or self.type == MatKind.diffuse_transmit or self.type == MatKind.coated_diffuse \
            or self.type == MatKind.conductor or self.type == MatKind.measured or self.type == MatKind.hair \
            or self.type == MatKind.coated_conductor

    @always_inline
    def has_lobe(self) -> Bool:
        """False when the lobe has no data: a measured material whose table did not load."""
        return not (self.type == MatKind.measured and self.measured_idx < Int32(0))

    @always_inline
    def lobe_param(self) -> Float32:
        """The lobe's scalar: the isotropic GGX alpha of a conductor or rough dielectric, else 0."""
        if self.type == MatKind.conductor or self.type == MatKind.dielectric:
            return max(self.roughU, self.roughV)
        return Float32(0)

    @always_inline
    def coat_alpha(self) -> Float32:
        """A layered material's coat roughness; a coated conductor keeps its metal's in roughU/V."""
        return self.emission.b if self.type == MatKind.coated_conductor else max(self.roughU, self.roughV)

    @always_inline
    def is_mirror(self) -> Bool:
        """A smooth conductor: its lobe samples a delta event and stores no vertex."""
        return self.type == MatKind.conductor and max(self.roughU, self.roughV) <= Float32(0.001)

    @always_inline
    def is_rough_dielectric(self) -> Bool:
        """pbrt's DielectricBxDF over a Trowbridge-Reitz microsurface: an ordinary glossy vertex, left to sampling,
        NEE and MIS, which MNEE, SMS and the specular-chain logic must not treat as glass."""
        return self.type == MatKind.dielectric and max(self.roughU, self.roughV) >= ALPHA_EFFECTIVELY_SMOOTH

    @always_inline
    def is_specular_glass(self) -> Bool:
        """Glass a specular chain passes through: a smooth or a thin dielectric. What MNEE and SMS walk."""
        return (self.type == MatKind.dielectric and not self.is_rough_dielectric()) or self.type == MatKind.thin_dielectric

    @always_inline
    def passes_rays(self) -> Bool:
        """A null interface or a Cycles lamp: rays continue through it unchanged."""
        return self.type == MatKind.interface or self.type == MatKind.lamp

    @always_inline
    def dt_reflectance(self, tex_rgb: RGB) -> RGB:
        """A diffusetransmission material's reflected colour; `tex_rgb` is its texture at the hit."""
        if Int(self.tex_idx) == -1 or self.dt_tex_lobes == Int8(2):
            return self.albedo
        return tex_rgb

    @always_inline
    def dt_transmittance(self, tex_rgb: RGB) -> RGB:
        """... and its transmitted colour."""
        if Int(self.tex_idx) == -1 or self.dt_tex_lobes == Int8(1):
            return self.emission
        return tex_rgb

    def glossy_filtered(self, min_ray_pdf: Float32, filter: Float32) -> Material:
        """Cycles' Filter Glossy: after a scatter of pdf p, glossy lobes widen to roughness sqrt(1 - p / filter) / 2.
        Hair takes it in emission.b (see _hair_precompute), microfacet materials as a floor on their alpha."""
        if filter <= Float32(0) or min_ray_pdf >= filter:
            return self
        var r = sqrt(Float32(1) - min_ray_pdf / filter) * Float32(0.5)
        var m = self
        if m.type == MatKind.hair:
            m.emission.b = r
        elif m.type == MatKind.conductor or m.type == MatKind.coated_diffuse or m.type == MatKind.coated_conductor or \
             (m.type == MatKind.dielectric and m.sss_boundary == Int8(0)):
            m.roughU = max(m.roughU, r)
            m.roughV = max(m.roughV, r)
        return m

    # A coated diffuse with sss_walk keeps the walk's parameters in the checker fields, which only a procedural
    # checkerboard uses: checker_tex1 = mean free path, checker_tex2 = (anisotropy, the walk's share of the base,
    # 1 for Cycles' van de Hulst albedo mapping or 0 for its Legacy fit).
    @always_inline
    def sss_walk_mfp(self) -> RGB:
        return self.checker_tex1

    @always_inline
    def sss_walk_g(self) -> Float32:
        return self.checker_tex2.r

    @always_inline
    def sss_walk_weight(self) -> Float32:
        return self.checker_tex2.g if self.sss_walk != Int8(0) else Float32(0)

    @always_inline
    def sss_walk_van_de_hulst(self) -> Bool:
        return self.checker_tex2.b != Float32(0)

    # A dielectric with sss_walk bounds a pbrt subsurface interior: checker_tex1 = extinction, checker_tex2 = single
    # scattering albedo, checker_uscale = anisotropy.
    @always_inline
    def sss_walk_interior(self) -> Bool:
        return self.type == MatKind.dielectric and self.sss_walk != Int8(0)

    @always_inline
    def mix_pick(self, u: Float32) -> Int:
        """A mix's sub-material index for a uniform draw `u`."""
        return Int((self.tex_idx >> 16) & Int32(0xFFFF)) if u < self.roughU else Int(self.tex_idx & Int32(0xFFFF))


@always_inline
def resolve_mix(materials: Pointer[Material, MutUntrackedOrigin], mut pcg: PCG32, mut mat: Material, mut mat_idx: Int):
    """A mix becomes one of its sub-materials, a mix of mixes diffuse; draws only for a mix."""
    if mat.type != MatKind.mix:
        return
    mat_idx = mat.mix_pick(pcg.next_float())
    mat = materials[unsafe_offset=mat_idx]
    if mat.type == MatKind.mix:
        mat.type = MatKind.diffuse

# ── Measured (tabulated) BRDF ────────────────────────────────────────────────
# One instance per distinct ".bsdf" tensor file (deduped by path — a scene may
# reference the same file from many materials, e.g. sportscar's car-paint
# body). Holds the real Dupuy & Jakob PiecewiseLinear2D marginal/conditional
# distributions for vndf/luminance (with derived CDFs) plus the
# Evaluate-only ndf/sigma/spectra tables — see measured_bsdf.mojo's loader
# for construction and bxdf.mojo's bxdf_{eval,sample,pdf}_measured for the
# consumers. Same struct reused for both the CPU host-pointer instance
# (built by the loader) and the GPU device-pointer instance (built in
# gpu.mojo's scene upload, Stage 3) — mirrors TriangleMesh/Curve's
# existing host-then-device-pointer-reuse convention.
#
# CAUTION: this struct has many pointer fields, so it must NEVER be passed BY
# VALUE across a real (non-inlined) function-call boundary -- see
# spectrum.mojo's long comment on the suspected Mojo miscompilation class
# (modular/modular#6759, later retracted by its own author as
# unreproducible; kept as a defensive workaround regardless). Always load a
# local `var mb = ...[idx]` and only hand it to @always_inline helpers.
@fieldwise_init
struct MeasuredBRDF(TrivialRegisterPassable):
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

# pbrt's TrowbridgeReitzDistribution::EffectivelySmooth threshold: below it a
# microfacet lobe is sampled as the delta it has become.
comptime ALPHA_EFFECTIVELY_SMOOTH = Float32(1e-3)


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
        eta = fast_recip(eta)
        cos_theta_i = -cos_theta_i
    var sin2_theta_i = max(Float32(0.0), Float32(1.0) - cos_theta_i * cos_theta_i)
    var sin2_theta_t = fast_div(sin2_theta_i, eta * eta)
    if sin2_theta_t >= Float32(1.0):
        return Float32(1.0)   # total internal reflection
    var cos_theta_t = safe_sqrt(Float32(1.0) - sin2_theta_t)
    var r_parl = fast_div(eta * cos_theta_i - cos_theta_t, eta * cos_theta_i + cos_theta_t)
    var r_perp = fast_div(cos_theta_i - eta * cos_theta_t, cos_theta_i + eta * cos_theta_t)
    return (r_parl * r_parl + r_perp * r_perp) * Float32(0.5)


@always_inline
def sss_boundary_lobe(cos_theta: Float32, eta: Float32) -> Float32:
    """A subsurface boundary's cosine lobe, Ft(cos)/c: pbrt's reciprocal S_w, normalised over the hemisphere."""
    var x = Float32(1) / eta
    var moment = Float32(0.45966) + x * (Float32(-1.73965) + x * (Float32(3.37668) + x * (Float32(-3.904945)
                 + x * (Float32(2.49277) + x * Float32(-0.68441)))))
    return (Float32(1) - fr_dielectric(cos_theta, eta)) / (Float32(1) - Float32(2) * moment)
