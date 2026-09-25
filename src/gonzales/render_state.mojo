"""Render-pipeline state structs, split out of geometry.mojo (the
per-cluster module split; see project_geometry_module_split memory).
FilmDims/FilterParams, and PathState through TileResult, were two
separate ranges in geometry.mojo. Depends on the core
(Point3f/Vec3f/RGB/SpectralSample/SampledWavelengths) and on primitives.mojo
(PathState holds a Ray) -- the one real cross-cluster dependency a
symbol-reference scan of the whole file found before this split, everywhere
else was comments matching a name."""
from gonzales.spectrum import SpectralSample, SampledWavelengths
from .geometry import Point3f, Vec3f, RGB
from .primitives import Ray

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


# ── Path state ────────────────────────────────────────────────────────────────
# See: docs/07_path_tracing.md

@fieldwise_init
struct PathState(TrivialRegisterPassable):
# <<listing: PathState>>
    var ray: Ray
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
    # Solid-angle pdf that THIS path's last scattering material would have
    # assigned, via its own env-light NEE sampler, to the direction it
    # actually scattered into -- the MIS partner for an escape that leaves
    # the scene and hits an untextured infinite light.
    #
    # It has to be carried rather than recomputed at the escape, because
    # there are TWO env NEE samplers and the miss handler cannot tell which
    # one ran: `diffuse` samples the env cosine-weighted (_nee_infinite_light,
    # a deliberate diffuse-only optimization), every other material samples it
    # uniformly over the sphere (_sample_infinite_light_nee, pdf INV_FOUR_PI).
    # The miss handler used to hardcode the cosine case as `pdf_light =
    # pdf_bsdf`, which is exact for diffuse -- cosine NEE and cosine BSDF
    # sampling really do share a pdf -- and wrong for everyone else, forcing
    # the power heuristic to a flat 0.5 on every escape while their NEE
    # partner kept using the true INV_FOUR_PI. The two strategies then no
    # longer partitioned unity. A near-mirror conductor has ~no response
    # toward a uniform-sphere direction, so its NEE contributed nothing and
    # the escape alone carried 0.5: the furnace test read 0.5023 where energy
    # conservation demands 1.0, and diffusetransmission read 0.6766.
    # Textured envs are unaffected -- the miss handler overrides this from
    # the CDF, which is the pdf both samplers share once one exists.
    var lastEnvNeePdf: Float32
    # Path length along the camera ray's SPECULAR chain, for the texture/bump
    # footprint (footprint.mojo): the ray cone standing in for pbrt's ray
    # differentials, 1 float instead of a full differential's 12. -1 once a
    # non-specular scatter has happened: pbrt then drops the differentials and
    # uses Camera::Approximate_dp_dxy at every later hit.
    var cone_len: Float32
# <</listing>>
# PathState layout: 24+12+12+12+4+8+8+1+1+1+1+4+4+4+4+4+4+4+8+20+4+4+4 = 148 bytes (was 144 -- +4 for cone_len);
# size is computed via size_of[PathState]() everywhere (GPU buffer sizing included), not hardcoded.

# ── GPU / render pipeline helpers ─────────────────────────────────────────────

@fieldwise_init
struct GpuTexture(TrivialRegisterPassable):
    comptime FORMAT_F32 = 0   # data holds Float32 linear RGB
    comptime FORMAT_U8 = 1    # data holds UInt8, decoded to linear through lut
    var data: Pointer[UInt8, MutUntrackedOrigin]    # device pointer: full mip pyramid, contiguous
    var lut: Pointer[Float32, MutUntrackedOrigin]   # device pointer: 256-entry byte -> linear table (FORMAT_U8 only)
    var width: Int32                                     # level-0 width
    var height: Int32                                    # level-0 height
    var n_levels: Int32                                  # number of mip levels stored in `data` (>=1)
    var channels: Int32                                  # stored channels per texel: 1 (replicated to RGB) or 3
    var format: Int32                                    # FORMAT_F32 or FORMAT_U8

@fieldwise_init
struct NormalSlopeMap(TrivialRegisterPassable):
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
    var slopes: Pointer[Float32, MutUntrackedOrigin]  # 2 floats/texel, row-major
    var res:    Int32                                      # 0 => absent

@always_inline
def normal_slope_map_none() -> NormalSlopeMap:
    return NormalSlopeMap(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0))

@fieldwise_init
struct ShadowTask(TrivialRegisterPassable):
    """A deferred shadow ray with its pre-computed radiance contribution."""
    var origin: Point3f
    var direction: Vec3f
    var tmax: Float32
    var contrib: SpectralSample   # deferred TRANSPORT, so spectral like throughput
    var active: Int32
    var _pad: Int32

@fieldwise_init
struct TileResult(TrivialRegisterPassable):
    var estimate: RGB
    var albedo: RGB
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32


# PathState.lastBsdfPdf sentinel: "a real scatter happened here, but its
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
