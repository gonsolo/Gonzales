# BDPT/VCM path vertex and the shared depth/storage limits.
# Part of the BDPT/VCM machinery that used to be one file (bdpt.mojo).

from .geometry import RGB, Point3f, Vec3f
from .render_state import PDF_VOL_PHASE_HIT
from .materials import LobeKind
from .spectrum import SampledWavelengths, SpectralSample

# last_bsdf_pdf sentinel: the previous camera-path event was a VOLUME scatter.
# Distinct from -1 (delta bounce, no competing strategy anywhere) because a
# volume vertex DOES have a competing strategy for a subsequent area-light
# hit -- its own s=1 connection to the light-source vertex -- and the hit must
# be MIS-weighted against it with the isotropic phase pdf 1/(4pi). It stays
# negative so the infinite-light miss handler keeps giving full weight: volume
# vertices do no environment NEE in this integrator, so nothing competes there.
comptime _VOL_PHASE_HIT: Float32 = PDF_VOL_PHASE_HIT   # geometry.mojo owns the sentinel space
comptime _BDPT_MAX_DEPTH = 40  # max surface/medium interactions per subpath (incl.
                                # non-stored delta/dielectric bounces — glass-of-water's
                                # nested water/ice/glass interfaces need ~30 crossings
                                # just to reach a real (diffuse) vertex)
comptime _BDPT_MAX_VERTS = 10  # max non-delta vertices per subpath. NOT light-only,
                                # despite the name: it caps how many vertices a light
                                # subpath stores in the shared cache (_bdpt_light_path_bounce)
                                # AND hard-terminates the CAMERA subpath at the same count
                                # (_bdpt_camera_path_bounce's `n_verts >= _BDPT_MAX_VERTS:
                                # return False`). So it bounds camera path LENGTH too --
                                # raising it changes image energy in multi-bounce scenes,
                                # it is not purely a cache-memory knob. Volume-scatter
                                # events store no vertex and so do NOT count against it;
                                # they consume _BDPT_MAX_DEPTH loop iterations instead.
                                # See project_photon_estimator_energy_gap /
                                # project_sphere_light_nee_bug memories for the measured
                                # effect of raising each. The camera-side
                                # volume branch deliberately does NOT increment
                                # n_verts -- re-adding that increment caps a
                                # dense-medium walk at 10 scatters and halves
                                # nothing visibly on thin media while costing
                                # ~2x on thick ones (0.45x -> 0.99x vs the path
                                # tracer when removed).
comptime _MNEE_MAX_SPHERES = 4  # cap on sphere-light MNEE call sites, unrolled via `if`
                                  # guards instead of a `for` loop -- see
                                  # _bdpt_mnee_sphere_light's own docstring for the real
                                  # GPU codegen bug (CUDA_ERROR_ILLEGAL_ADDRESS) this
                                  # works around. Scenes with more emissive spheres than
                                  # this only lose MNEE's glass-behind-sphere handling for
                                  # the extras -- ordinary NEE/connect/merge still reaches
                                  # them normally.

# ── VCM (Vertex Connection and Merging, Georgiev et al. 2012) ────────────────
# Real VCM combines vertex CONNECTION (_bdpt_connect_to_cache/_connect) and
# vertex MERGING (_bdpt_merge_from_cache) by running BOTH, unconditionally,
# at every non-delta camera vertex, and summing their contributions -- NOT a
# stochastic either/or pick (that was this codebase's Stage 1 design,
# retired VCM Stage 2c/2d; see project_vcm_stage2_mis_derivation memory and
# git history for why: SmallVCM's own reference driver loop
# (vertexcm.hxx's PathTracerEyeVertex, ConnectVertices + the RangeQuery
# grid walk) does exactly this -- two separate loops per eye vertex, no
# selection probability anywhere). Each technique's own per-candidate MIS
# weight (Georgiev Eq. 9-10 / SmallVCM's ConnectVertices and
# RangeQuery::Process, both verified against the reference source) already
# makes the UNWEIGHTED SUM of both techniques' outputs a correct, lower-
# variance combined estimator -- no rescaling by any selection probability
# is needed or correct here.
#
# Both techniques' weights are real for diffuse/conductor/coated_conductor/
# hair/measured (mat_kind 0/1/2/3, real surface) cv/lv pairs -- everything
# with a genuine standalone BSDF pdf, see _bdpt_vertex_pdfs. Only volume
# (isotropic phase, no surface normal) and dielectric/thin_dielectric
# (genuinely delta/specular, never even stored as LVC vertices) fall
# through to weight=1 / are unreachable here, a deliberately scoped
# architectural boundary, not a silent omission -- see _connect's and
# _bdpt_merge_from_cache's own docstrings for the exact scope condition.
#
# The merge radius is now progressive (Stage 2c, see _bdpt_render_core's
# per-sample loop): a single global radius r_i = r_1/(i+1)^(0.5*(1-alpha))
# shrinks every spp sample (Hachisuka & Jensen 2008's iteration-indexed
# scheme, NOT sppm.mojo's per-pixel Knaus-Zwicker adaptive radius -- the
# two are architecturally different and not interchangeable, see that
# file's _sppm_gather_one for contrast). This is what makes enabling real
# weighted merging safe: a FIXED radius merge is only CONSISTENT (converges
# to zero bias as radius->0), never unbiased at any one radius, and a fixed
# small radius on a freshly-rebuilt-every-sample LVC (no cross-pass photon
# accumulation) produces classic single-shot photon-mapping fireflies --
# confirmed by this session's Stage 1 predecessor. Progressive shrinkage
# fixes that the same way SPPM's own progressive radius does.

# ── Light Vertex Cache (LVC-BPT, Davidovic et al. 2014, restructured VCM ─────
# Stage 2b for standard Veach pairing) ────────────────────────────────────────
# One light subpath is traced per pixel (`n_light_paths == n_pix`), each into
# its own dedicated slice of a shared `lvc` buffer (see
# _bdpt_store_lvc_vertex's docstring) — standard Veach BDPT pairing, not a
# shared-pool random-draw (that was this codebase's original LVC-BPT design;
# replaced because Georgiev/SmallVCM's real per-vertex MIS weights assume
# per-pixel-paired light subpaths, see project_vcm_stage2_mis_derivation
# memory). Each pixel's camera subpath connects to EVERY vertex of its own
# paired light path (`_bdpt_connect_to_cache`) and merges against ALL light
# paths' vertices via a shared spatial grid (`_bdpt_merge_from_cache`) — see
# `_bdpt_trace_light_path`/`_bdpt_trace_camera_and_connect` below, both
# `comptime[use_gpu: Bool]` parameterized so CPU and GPU share one
# implementation.

# ── Vertex types ──────────────────────────────────────────────────────────────

@fieldwise_init
struct BDPTVertex(TrivialRegisterPassable):
    """A vertex on a camera or light subpath."""
    var pos:    Point3f  # world position
    # THE GEOMETRIC normal (0 for volume). Keep it geometric: it is what
    # _connect's solid-angle -> area pdf conversions (pbrt's ConvertDensity)
    # are built on, and a perturbed normal there is a real bias, not a
    # shading choice. See shading_normal below.
    var normal: Vec3f
    # The SHADING normal -- `normal` after bump/normal maps. Only the BxDF
    # interface reads it (via _vertex_ctx), which is the split pbrt keeps as
    # Vertex::ng vs Vertex::ns. A connection's cosine at each endpoint comes
    # out of that BxDF evaluation (f_cos), so this is also the normal that
    # cosine is taken against.
    #
    # This field exists because of a measured mistake: the first version of
    # the light/photon-side surface-map work wrote the PERTURBED normal into
    # `normal` alone, so every connection's G and its two area-pdf
    # conversions silently used it. The signature was VCM reacting to a
    # normal map about twice as strongly as the path tracer (barcelona whole
    # image 0.9009 -> 0.8498 against a pbrt BDPT reference, i.e. moving away
    # from the path tracer's 0.9851, where a shading-only change should have
    # moved it toward). Two cosines per connection, each wrong, is exactly a
    # factor of two.
    #
    # Set it at EVERY store site, including the ones with no map, where it is
    # simply equal to `normal` -- a vertex left with the _null_vertex default
    # here is shaded against a normal that has nothing to do with its surface.
    var shading_normal: Vec3f
    var beta: SpectralSample  # throughput to here, at THIS PASS's hero wavelengths
    var alb:  RGB  # BSDF albedo (F0 for conductor)
    var pdf_fwd: Float32  # area PDF forward (from previous vertex) -- unused by the
                           # dVCM/dVC/dVM MIS scheme below (kept for other callers)
    var pdf_bwd: Float32  # repurposed to hold the isotropic GGX alpha for mat_kind=1
                           # (conductor) vertices -- NOT a Veach reverse-pdf
    # VCM Stage 2b (2026-07-10): real per-vertex MIS quantities, ported
    # verbatim from Georgiev et al. 2012 ("Light Transport Simulation with
    # Vertex Connection and Merging") / the SmallVCM reference
    # implementation (github.com/SmallVCM/SmallVCM, src/vertexcm.hxx) --
    # see project_vcm_stage2_mis_derivation memory for the full verified
    # formulas this session grounded against the actual paper + that code.
    # Recursively updated at every bounce on both light and camera
    # subpaths; consumed by `_connect`'s and `_bdpt_merge_from_cache`'s MIS
    # weights. Do NOT hand-derive these from scratch -- follow the memory's
    # verbatim formulas; getting this wrong silently biases the image.
    var dVCM: Float32  # MIS quantity used for BOTH connection and merging
    var dVC:  Float32  # MIS quantity used for vertex connection
    var dVM:  Float32  # MIS quantity used for vertex merging
    var is_surface: Int32  # 1 = surface hit, 0 = volume scatter
    var is_delta:   Int32  # 1 = specular (mirror conductor / dielectric) — cannot be connected
    var is_light:   Int32  # 1 = this is a light-source vertex (s=0 in BDPT notation)
    var med_idx:    Int32  # medium index AFTER this vertex (-1 = vacuum)
    var mat_kind:   Int32  # a LobeKind
    var n_delta:    Int32  # delta (specular) bounces on its subpath before it; they count toward maxdepth
    # Direction back toward this vertex's own predecessor on its subpath
    # (-incoming ray direction). Populated for mat_kind=1 (GGX needs both
    # directions around the half-vector) and mat_kind=2 (hair's wo, needed to
    # recompute HairLobeConstants via _hair_precompute at eval time).
    var wo: Vec3f
    # mat_kind=2 (hair) only: material index (to re-fetch eta/sigma_a/betaM/
    # betaN from sd.materials) + curve hit info (to re-derive the fiber frame
    # via _hair_precompute) — NOT stored inline as the full ~30-field
    # HairLobeConstants, to keep this struct small for every OTHER vertex
    # kind; recomputing per connection is the same cost class as
    # _eval_conductor_ggx's own per-call GGX evaluation.
    var mat_idx: Int32
    var hair_curve_idx: Int32
    var hair_h: Float32
    var hair_v: Float32
    # Hero-wavelength sample this vertex's subpath was traced at (staged
    # spectral rollout, see project_spectral_rendering memory /
    # lovely-dazzling-meteor plan). Unused until Stage 3; needed on every
    # vertex (not just the subpath root) because the Light-Vertex-Cache
    # connects camera vertices to globally-random-indexed light vertices —
    # no natural per-path pairing to inherit wavelengths from.
    var wavelengths: SampledWavelengths

@always_inline
def _null_vertex() -> BDPTVertex:
    return BDPTVertex(
        pos=Point3f(Float32(0)),
        normal=Vec3f(Float32(0), Float32(1), Float32(0)),
        shading_normal=Vec3f(Float32(0), Float32(1), Float32(0)),
        beta=SpectralSample(Float32(0)),
        alb=RGB(Float32(0)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(0), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian, n_delta=Int32(0),
        wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

# ── Geometry helpers ──────────────────────────────────────────────────────────
