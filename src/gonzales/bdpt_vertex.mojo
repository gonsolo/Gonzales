# BDPT/VCM path vertex and the shared depth/storage limits.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from .geometry import RGB, Point3f, Vec3f
from .render_state import PDF_VOL_PHASE_HIT
from .materials import LobeKind
from .spectrum import SampledWavelengths, SpectralSample

# last_bsdf_pdf sentinel: the previous camera-path event was a VOLUME scatter.
# Distinct from -1 (delta bounce, no competing strategy anywhere) because a
comptime _VOL_PHASE_HIT: Float32 = PDF_VOL_PHASE_HIT   # geometry.mojo owns the sentinel space
comptime _BDPT_MAX_DEPTH = 40  # max surface/medium interactions per subpath (incl.
                                # non-stored delta/dielectric bounces — glass-of-water's
                                # nested water/ice/glass interfaces need ~30 crossings
                                # just to reach a real (diffuse) vertex)
comptime _BDPT_MAX_VERTS = 10  # max non-delta vertices per subpath. NOT light-only,
                                # despite the name: it caps how many vertices a light
                                # subpath stores in the shared cache (_bdpt_light_path_bounce)
comptime _MNEE_MAX_SPHERES = 4  # cap on sphere-light MNEE call sites, unrolled via `if`
                                  # guards instead of a `for` loop -- see
                                  # _bdpt_mnee_sphere_light's own docstring for the real

# ── VCM (Vertex Connection and Merging, Georgiev et al. 2012) ────────────────
# Real VCM combines vertex CONNECTION (_bdpt_connect_to_cache/_connect) and
# vertex MERGING (_bdpt_merge_from_cache) by running BOTH, unconditionally,

# ── Light Vertex Cache (LVC-BPT, Davidovic et al. 2014, restructured VCM ─────
# Stage 2b for standard Veach pairing) ────────────────────────────────────────
# One light subpath is traced per pixel (`n_light_paths == n_pix`), each into
# its own dedicated slice of a shared `lvc` buffer (see

# ── Vertex types ──────────────────────────────────────────────────────────────

@fieldwise_init
struct BDPTVertex(TrivialRegisterPassable):
    """A vertex on a camera or light subpath."""
    var pos:    Point3f  # world position
    # THE GEOMETRIC normal (0 for volume). Keep it geometric: it is what
    # _connect's solid-angle -> area pdf conversions (pbrt's ConvertDensity)
    var normal: Vec3f
    # The SHADING normal -- `normal` after bump/normal maps. Only the BxDF
    # interface reads it (via _vertex_ctx), which is the split pbrt keeps as
    var shading_normal: Vec3f
    var beta: SpectralSample  # throughput to here, at THIS PASS's hero wavelengths
    var alb:  RGB  # BSDF albedo (F0 for conductor)
    var pdf_fwd: Float32  # area PDF forward (from previous vertex) -- unused by the
                           # dVCM/dVC/dVM MIS scheme below (kept for other callers)
    var pdf_bwd: Float32  # repurposed to hold the isotropic GGX alpha for mat_kind=1
                           # (conductor) vertices -- NOT a Veach reverse-pdf
    # VCM Stage 2b (2026-07-10): real per-vertex MIS quantities, ported
    # verbatim from Georgiev et al. 2012 ("Light Transport Simulation with
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
    var wo: Vec3f
    # mat_kind=2 (hair) only: material index (to re-fetch eta/sigma_a/betaM/
    # betaN from sd.materials) + curve hit info (to re-derive the fiber frame
    var mat_idx: Int32
    var hair_curve_idx: Int32
    var hair_h: Float32
    var hair_v: Float32
    # Hero-wavelength sample this vertex's subpath was traced at (staged
    # spectral rollout, see project_spectral_rendering memory /
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
