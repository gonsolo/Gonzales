# Bidirectional Path Tracing (CPU, multithreaded, + GPU).
# Supports homogeneous participating media and specular chains (glass).
# Strategies: t >= 1, s >= 1 only (no lens sampling for s=0).
# MIS: balance heuristic over all valid connection strategies.

from std.sys import has_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.algorithm import parallelize
from std.math import sqrt, cos, sin, tan, floor, log, exp, max, abs, ceildiv, pow
from std.memory import alloc
from std.atomic import Atomic
from .geometry import (
    RGB, SampledSpectrum, Point3f, Point2f, Vec3f, vec3f, point3f, Ray_C, Intersection_C, Frame,
    TriangleMesh_C, Material_C, MatKind, AreaLight_C, Medium_C, MediumInterface_C,
    Sphere_C, Curve_C, PrimId_C, Instance_C, DistantLight_C, InfiniteLight_C, PointLight_C,
    MeasuredBRDF_C, GpuTexture_C,
    dot, cross, fr_dielectric, sphere_outward_normal, PI, INV_FOUR_PI, INV_PI,
)
from .bvh import (
    BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core, test_spheres, _mk_sd_full,
    _scene_bounding_sphere, _sample_disk_perpendicular, _sample_infinite_light_dir, _eval_infinite_light_and_pdf,
    HairLobeConstants, _hair_precompute, _hair_eval_lobes, _hair_sample_dir, curve_offset_eps,
    LightSample, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, _sample_infinite_light_nee,
    render_aux_buffers,
)
from .sampling import power_heuristic
from .rng import PCG32
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image, denoise
from .sppm import _geom_normal, _dielectric_bounce, _sppm_update_medium, _cosine_hemisphere_sample, sample_homogeneous_free_flight, sample_area_light_uniform, sppm_reset_i32_gpu, _HSIZE, _hash_cell, _sppm_render_core
from .shading import _tex_lookup, _get_tri_verts
from .bxdf import GeomContext, BxDFSample, bxdf_sample_conductor, bxdf_sample_coated_conductor, bxdf_is_delta, bxdf_eval_diffuse, bxdf_pdf_diffuse, ggx_D, ggx_G2, bxdf_eval_conductor_ggx, bxdf_pdf_conductor_ggx, _nee_weight_simple, _nee_weight_hair, _nee_weight_simple_via_spectral
from .measured_bxdf_eval import bxdf_eval_measured, bxdf_sample_measured, _nee_weight_measured, bxdf_pdf_measured
from .gpu import GpuSceneHandle
from .spectrum import (
    SampledWavelengths, SpectralSample, sample_wavelengths_uniform,
    rgb_to_spectral_sample, rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb,
)

comptime _BDPT_MAX_DEPTH = 40  # max surface/medium interactions per subpath (incl.
                                # non-stored delta/dielectric bounces — glass-of-water's
                                # nested water/ice/glass interfaces need ~30 crossings
                                # just to reach a real (diffuse) vertex)
comptime _BDPT_MAX_VERTS = 10  # max non-delta vertices stored per light subpath (caps
                                # each light path's contribution to the shared cache below)

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
# Both techniques' weights are scoped to diffuse (mat_kind=0, real surface)
# cv/lv pairs only this pass -- conductor/hair/measured have no real
# standalone BSDF pdf yet (or, for measured, a real pdf but no connect-time
# reverse-pdf local frame wired up), so connections/merges touching them
# fall through to weight=1 (today's plain unweighted behavior), a
# deliberately scoped gap tracked in project_vcm_stage2_mis_derivation
# memory, not a silent omission. See _connect's and
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
    var normal: Vec3f    # geometric normal (0 for volume)
    var beta: RGB  # throughput to here
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
    var mat_kind:   Int32  # 0 = Lambertian (diffuse/volume), 1 = rough conductor (GGX), 2 = hair (Marschner 3-lobe)
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
        beta=RGB(Float32(0)),
        alb=RGB(Float32(0)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(0), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=Int32(0),
        wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

# ── Geometry helpers ──────────────────────────────────────────────────────────

@always_inline
def _bdpt_medium_update(
    ray_dir: SIMD[DType.float32, 3],
    inter:   Intersection_C,
    mat:     Material_C,
    sd:      SceneDescriptor2_C,
    hit: Point3f,
) -> Int32:
    """Determine new current_medium_idx after crossing a surface."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return Int32(-1)
    var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
    var n: Vec3f
    if inter.primId.type == Int8(4):
        # Analytic sphere: outward normal = normalize(hit - center)
        var si = Int(inter.primId.id1)
        var sph = sd.spheres[si]
        n = sphere_outward_normal(hit, sph.center)
    else:
        var mi: Int; var bv: Int
        if inter.primId.type == 0:
            mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
        else:
            mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        var m  = sd.meshes[mi]
        var v0 = Int(m.vertexIndices[bv]); var v1 = Int(m.vertexIndices[bv+1]); var v2 = Int(m.vertexIndices[bv+2])
        var p0 = Point3f(m.points[v0*4], m.points[v0*4+1], m.points[v0*4+2])
        var p1 = Point3f(m.points[v1*4], m.points[v1*4+1], m.points[v1*4+2])
        var p2 = Point3f(m.points[v2*4], m.points[v2*4+1], m.points[v2*4+2])
        var e1 = p1 - p0; var e2 = p2 - p0
        n = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
    var md = ray_dir[0]*n.x + ray_dir[1]*n.y + ray_dir[2]*n.z
    return iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx

# ── Visibility with transmittance ─────────────────────────────────────────────

def _visible_transmittance(
    a: Point3f, b: Point3f,
    med_idx: Int32,
    sd:      SceneDescriptor2_C,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
) -> SIMD[DType.float32, 3]:
    """Returns transmittance along segment AB, or (0,0,0) if occluded.
    Glass (dielectric) surfaces are passed through with Fresnel transmittance.
    `scratch` is one caller-owned Intersection_C slot (no internal alloc/free)
    so this is safe to call from a GPU kernel thread — every existing GPU
    kernel in this codebase takes pre-allocated, thread-indexed scratch
    instead of allocating per-thread (see sppm_gen_vp_gpu's inter_scratch)."""
    var d = b - a
    var dist_total = d.length()
    if dist_total < Float32(1e-5):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var inv = Float32(1) / dist_total
    var dir = SIMD[DType.float32, 3](d.x*inv, d.y*inv, d.z*inv)

    var Tr = RGB(Float32(1))
    var org = a + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
    var remaining = dist_total - Float32(0.0002)
    var cur_med = med_idx

    var inter_mem = scratch
    for _ in range(8):
        if remaining < Float32(1e-4): break
        var ray = Ray_C(org, Vec3f(dir[0], dir[1], dir[2]))
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, remaining * Float32(0.9995), inter_mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        # test_spheres (analytic spheres, e.g. the caustic sphere) aren't part of
        # the BVH — traverse_bvh2_core only tests triangles/curves — so they need
        # a separate pass. Seed a sentinel tHit=remaining*0.9995 when the BVH found
        # nothing, so test_spheres's own internal tMax (it only bounds itself by
        # result[0].tHit when result[0].hit is already set) respects the shadow
        # ray's segment length instead of defaulting to unbounded (1e38).
        var had_bvh_hit = inter_mem[0].hit != Int8(0)
        if not had_bvh_hit:
            inter_mem[0].hit = Int8(1)
            inter_mem[0].tHit = remaining * Float32(0.9995)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, inter_mem)
        if not had_bvh_hit and inter_mem[0].primId.type != Int8(4):
            inter_mem[0].hit = Int8(0)
        if inter_mem[0].hit == Int8(0):
            # Nothing between here and destination: apply remaining Beer-Lambert
            if Int(cur_med) >= 0:
                var med = sd.mediums[Int(cur_med)]
                var sigma_t = med.sigma_a + med.sigma_s
                Tr *= RGB(exp(-sigma_t.r*remaining), exp(-sigma_t.g*remaining), exp(-sigma_t.b*remaining))
            break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = org + Vec3f(dir[0], dir[1], dir[2]) * t_hit

        # Beer-Lambert through medium segment up to hit
        if Int(cur_med) >= 0:
            var med = sd.mediums[Int(cur_med)]
            var sigma_t = med.sigma_a + med.sigma_s
            Tr *= RGB(exp(-sigma_t.r*t_hit), exp(-sigma_t.g*t_hit), exp(-sigma_t.b*t_hit))

        if mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # Pass through glass with Fresnel transmittance
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var facing = dot(dir, gn) < Float32(0)
            var n_for_cos = gn if facing else gn*Float32(-1)
            var cos_i = -dot(dir, n_for_cos)
            if cos_i < Float32(0): cos_i = -cos_i
            var ior = mat.albedo.r
            var fr = fr_dielectric(cos_i, Float32(1)/ior if facing else ior)
            var T = Float32(1) - fr
            Tr *= T
            if Tr.r < Float32(1e-7) and Tr.g < Float32(1e-7) and Tr.b < Float32(1e-7):
                return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
            # Update medium after crossing glass surface
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var md = dir[0]*gn[0]+dir[1]*gn[1]+dir[2]*gn[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        elif mat.type == MatKind.interface:
            # Pure medium boundary: update medium, continue
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var igna = _geom_normal(inter, sd.meshes, sd.instances)
                var md = dir[0]*igna[0]+dir[1]*igna[1]+dir[2]*igna[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        else:
            # Opaque surface blocks the segment
            return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    return SIMD[DType.float32, 3](Tr.r, Tr.g, Tr.b)

@always_inline
def _bdpt_nee_contribute(
    beta: RGB,
    w: RGB,
    ls: LightSample,
    hit: Point3f,
    gn: SIMD[DType.float32, 3],
    cur_med_idx: Int32,
    sd: SceneDescriptor2_C,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    eps: Float32 = Float32(0.0001),
) -> RGB:
    """BDPT-side NEE glue shared by every per-material light loop below:
    given a LightSample + material weight (from the shared Light interface
    — bvh.mojo's LightSample samplers — and BxDF interface —
    bxdf.mojo's _nee_weight_simple/_nee_weight_hair), test transmittance
    (BDPT's own occlusion primitive, media-aware — unlike shading.mojo/
    sppm.mojo's boolean any-hit test, so this stays a BDPT-local helper
    rather than a fully cross-integrator one) and return the beta-weighted
    contribution, or black if invalid/occluded. `eps` defaults to the fixed
    offset used for triangle/sphere hits; hair call sites pass
    curve_offset_eps(hc.radius) instead (see bvh.mojo)."""
    if w.is_black():
        return RGB(Float32(0))
    var shadow_org = hit + vec3f(gn) * eps
    var shadow_end = shadow_org + Vec3f(ls.wi[0], ls.wi[1], ls.wi[2]) * ls.dist
    var Tr = _visible_transmittance(shadow_org, shadow_end, cur_med_idx, sd, scratch)
    if Tr[0] > Float32(0) or Tr[1] > Float32(0) or Tr[2] > Float32(0):
        return beta * w * RGB(Tr[0], Tr[1], Tr[2])
    return RGB(Float32(0))

# ── Cosine-area PDF conversion ────────────────────────────────────────────────

@always_inline
def _pdf_solid_to_area(pdf_solid: Float32, cos_theta: Float32, dist2: Float32) -> Float32:
    """Convert solid-angle PDF to area PDF: p_A = p_ω * |cosθ| / r²."""
    if dist2 < Float32(1e-8): return Float32(0)
    return pdf_solid * (cos_theta if cos_theta > Float32(0) else -cos_theta) / dist2

# ── Store a vertex in the shared Light Vertex Cache ──────────────────────────

@always_inline
def _bdpt_store_lvc_vertex(
    v: BDPTVertex,
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lp_idx: Int,
    local_idx: Int,
):
    """Store vertex `local_idx` (0-indexed, always < _BDPT_MAX_VERTS by
    construction of the caller's loop bound) of light path `lp_idx` into
    its own dedicated slice of the LVC: light path `lp_idx` owns exactly
    the slots [lp_idx*_BDPT_MAX_VERTS, (lp_idx+1)*_BDPT_MAX_VERTS).
    VCM Stage 2b (2026-07-10): replaced the old shared-global-cache +
    atomic-slot-reservation design (every light path competing for slots in
    one flat array) with this per-path-indexed layout, needed so the
    camera side can deterministically pair each pixel with its OWN light
    path (real Georgiev/SmallVCM-style VCM's dVCM/dVC/dVM MIS weights
    assume that pairing, not a random shared-pool draw — see
    project_vcm_stage2_mis_derivation memory). Bonus: since each light
    path now owns a non-contended slice, no atomics are needed here at
    all, on CPU or GPU."""
    lvc[lp_idx * _BDPT_MAX_VERTS + local_idx] = v

# ── LVC connection scale factor ──────────────────────────────────────────────

@always_inline
def _bdpt_connect_to_cache(
    cv: BDPTVertex,
    sd: SceneDescriptor2_C,
    has_med: Bool,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lp_idx: Int,
    path_len: Int,
    mis_vm_weight_factor: Float32,
) -> RGB:
    """VCM Stage 2b (2026-07-10): connect eye vertex `cv` to EVERY vertex of
    its deterministically PAIRED light path (`lp_idx` — standard Veach BDPT
    pairing: n_light_paths == n_pix, one dedicated light path per pixel,
    see _bdpt_store_lvc_vertex's docstring). This REPLACES the old
    K-uniformly-random-draws-from-a-shared-global-pool estimator (and its
    avg_light_path_len/K rescaling — no longer needed, since every vertex
    of the ONE paired path is visited exactly once, an exhaustive sum, not
    a subsample). Real VCM's dVCM/dVC/dVM MIS weights (see
    project_vcm_stage2_mis_derivation memory) assume exactly this pairing;
    the old random-subsample design was not verified compatible with them.
    No RNG needed here anymore — the set of light vertices to connect to is
    now fully determined by which pixel `cv`'s eye subpath belongs to."""
    var sum = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    for local in range(path_len):
        var lv = lvc[lp_idx * _BDPT_MAX_VERTS + local]
        sum += _connect(cv, lv, sd, has_med, scratch, mis_vm_weight_factor)
    return RGB(sum[0], sum[1], sum[2])

# ── VCM vertex merging: spatial hash grid over the LVC ───────────────────────
# Mirrors sppm.mojo's photon hash grid (_build_grid/_sppm_insert_photon/
# _sppm_reset_grid_cell) exactly, but keyed on BDPTVertex.pos instead of
# SPPMPhoton.pos, and using a SEPARATE parallel `merge_next` array for
# chaining rather than a field inside BDPTVertex itself (avoids touching
# BDPTVertex's layout/every other construction site in this file). Reuses
# sppm.mojo's _HSIZE bucket count and _hash_cell function directly -- no
# reason for the grid math itself to differ between the two use sites.

def _bdpt_reset_merge_cell(heads: UnsafePointer[Int32, MutAnyOrigin], h: Int):
    heads[h] = Int32(-1)

def _bdpt_insert_merge_vertex[use_gpu: Bool](
    k: Int,
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_path_len: UnsafePointer[Int32, MutAnyOrigin],
    merge_next: UnsafePointer[Int32, MutAnyOrigin],
    heads: UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    """Insert LVC slot `k` into the merge hash grid, unless it's an unused
    tail slot of its light path's per-path slice (VCM Stage 2b storage
    layout, see _bdpt_store_lvc_vertex's docstring — `k` ranges over the
    full `n_light_paths * _BDPT_MAX_VERTS` capacity, not just the vertices
    actually stored). Comptime-branches only on the bucket-head update
    primitive -- identical pattern to sppm.mojo's _sppm_insert_photon[use_gpu]."""
    var lp_idx = k // _BDPT_MAX_VERTS
    var local_idx = k % _BDPT_MAX_VERTS
    if local_idx >= Int(lvc_path_len[lp_idx]):
        return
    var ix = Int(floor(lvc[k].pos.x * inv_cell))
    var iy = Int(floor(lvc[k].pos.y * inv_cell))
    var iz = Int(floor(lvc[k].pos.z * inv_cell))
    var h = _hash_cell(ix, iy, iz)
    comptime if use_gpu:
        var old = Atomic._xchg(heads + h, Int32(k))
        merge_next[k] = old
    else:
        merge_next[k] = heads[h]
        heads[h] = Int32(k)

def _bdpt_build_merge_grid(
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_path_len: UnsafePointer[Int32, MutAnyOrigin],
    n_light_paths: Int,
    merge_next: UnsafePointer[Int32, MutAnyOrigin],
    heads: UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    """CPU-only grid build (mirrors sppm.mojo's _build_grid): reset all
    buckets, then insert every LVC vertex via the SAME atomic-exchange
    insert the GPU kernel uses (parallel CPU workers race on bucket heads
    exactly like GPU threads would). Iterates the full
    `n_light_paths * _BDPT_MAX_VERTS` capacity;
    `_bdpt_insert_merge_vertex` itself skips each path's unused tail slots
    via `lvc_path_len`."""
    @parameter
    def reset_one(i: Int):
        _bdpt_reset_merge_cell(heads, i)
    parallelize[reset_one](_HSIZE)

    @parameter
    def insert_one(k: Int):
        _bdpt_insert_merge_vertex[True](k, lvc, lvc_path_len, merge_next, heads, inv_cell)
    parallelize[insert_one](n_light_paths * _BDPT_MAX_VERTS)

@always_inline
def _bdpt_merge_from_cache(
    cv: BDPTVertex,
    sd: SceneDescriptor2_C,
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    merge_next: UnsafePointer[Int32, MutAnyOrigin],
    heads: UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
    r2: Float32,
    norm: Float32,
    mis_vc_weight_factor: Float32,
) -> RGB:
    """Vertex MERGING (photon-mapping-style density estimation) against the
    shared Light Vertex Cache -- the "M" in VCM, run UNCONDITIONALLY
    alongside _bdpt_connect_to_cache's vertex CONNECTION for every non-delta
    camera vertex (real VCM does both every time, not a stochastic either/or
    -- see _bdpt_trace_camera_and_connect's call sites).

    For each LVC vertex `lv` within radius sqrt(r2) of `cv`, treats `lv` as
    a stored photon: evaluates cv's own BSDF toward lv's stored `wo`
    (the direction the light path arrived from at lv -- exactly the
    `-photon.dir_in` convention sppm.mojo's _sppm_gather_one already uses,
    since BDPTVertex.wo IS that same "direction back toward the light"
    quantity for a light-subpath vertex), multiplies by lv's beta (already
    a valid, unbiased single-light-path throughput estimate -- same
    quantity _bdpt_connect_to_cache's _connect already uses for the light
    side), and divides by (n_light_paths * pi * r2): the standard photon-
    density-estimation normalization, where n_light_paths independent light
    subpaths are the "N emitted photons" and each contributes AT MOST the
    non-delta vertices it stored (mirroring _bdpt_lvc_connection_scale's
    own 1/n_light_paths derivation for connections -- see that function's
    docstring). No shadow ray, no geometry term: merging assumes cv and lv
    are close enough to be treated as the same point, so lv's own light
    path having reached lv unoccluded already implies the segment is
    clear.

    VCM Stage 2c/2d: real per-candidate MIS weight (Georgiev et al. 2012 /
    SmallVCM's RangeQuery::Process, vertexcm.hxx:129-166, verified against
    the reference source) is applied when both cv and lv are diffuse or
    rough conductor/coated_conductor (_bdpt_vertex_mis_scoped) -- the same
    scope _connect uses for its own connection weight, for the same reason
    (no real standalone pdf for hair/measured yet, see that function's
    docstring)."""
    if cv.is_delta != Int32(0) or cv.is_surface == Int32(0):
        return RGB(Float32(0))
    var total = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var cix = Int(floor(cv.pos.x * inv_cell))
    var ciy = Int(floor(cv.pos.y * inv_cell))
    var ciz = Int(floor(cv.pos.z * inv_cell))
    for ddx in range(-1, 2):
        for ddy in range(-1, 2):
            for ddz in range(-1, 2):
                var h = _hash_cell(cix + ddx, ciy + ddy, ciz + ddz)
                var k = Int(heads[h])
                while k != -1:
                    var lv = lvc[k]
                    # is_light==1 vertices are the light SOURCE's own point
                    # (the s=1 connection strategy): their beta is 1/pdf_area
                    # ONLY, with the actual emitted radiance held separately
                    # in lv.alb (see _connect's own is_light special case,
                    # which multiplies the two together). Merging with them
                    # using the generic "beta = flux" assumption below would
                    # silently drop that emission factor -- exactly matching
                    # why sppm.mojo's _sppm_trace_photon never stores a
                    # photon at bounce==0 either (its own docstring: "that
                    # direct contribution is now covered by NEE instead").
                    # Every OTHER stored vertex's beta already has emission
                    # folded in via the light path's own flux computation.
                    if lv.is_delta == Int32(0) and lv.is_surface == Int32(1) and lv.is_light == Int32(0):
                        var e = lv.pos - cv.pos
                        var dist2 = e.length_sq()
                        if dist2 <= r2:
                            var f_cv = _eval_vertex(cv, lv.wo.to_simd(), sd)
                            var w = Float32(1)
                            if _bdpt_vertex_mis_scoped(cv) and _bdpt_vertex_mis_scoped(lv):
                                var (camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w) = _bdpt_vertex_pdfs(cv, lv.wo.to_simd(), sd)
                                var w_light = lv.dVCM * mis_vc_weight_factor + lv.dVM * camera_bsdf_dir_pdf_w
                                var w_camera = cv.dVCM * mis_vc_weight_factor + cv.dVM * camera_bsdf_rev_pdf_w
                                w = Float32(1) / (w_light + Float32(1) + w_camera)
                            total[0] += f_cv[0] * lv.beta.r * w
                            total[1] += f_cv[1] * lv.beta.g * w
                            total[2] += f_cv[2] * lv.beta.b * w
                    k = Int(merge_next[k])
    var beta = cv.beta
    return RGB(total[0] * beta.r, total[1] * beta.g, total[2] * beta.b) * norm

# ── Trace one camera subpath, connecting to the shared cache inline ─────────

def _bdpt_trace_camera_and_connect[use_gpu: Bool](
    r2c:     UnsafePointer[Float32, MutAnyOrigin],
    c2w:     UnsafePointer[Float32, MutAnyOrigin],
    px:      Int, py:      Int,
    sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc:     UnsafePointer[BDPTVertex, MutAnyOrigin],
    lp_idx:  Int,
    path_len: Int,
    merge_next: UnsafePointer[Int32, MutAnyOrigin],
    merge_heads: UnsafePointer[Int32, MutAnyOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    px_scale: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_f: Float32,
    start_med_idx: Int32 = Int32(-1),
) -> Tuple[RGB, RGB]:
    """Trace one camera subpath from pixel (px,py). At each non-delta vertex,
    connect inline/synchronously to the shared Light Vertex Cache via
    `_bdpt_connect_to_cache` — mirrors how every live GPU shading kernel in
    this codebase already does its shadow ray (any_hit test, straight into
    the thread's own accumulator; gpu.mojo's queued ShadowTask_C mechanism
    is dead code, never used by the live render loop). Returns (total, first_alb):
    this camera path's total contribution for one spp sample, and the material
    albedo at its first non-delta (stored) vertex — the same "first hit,
    skipping through mirrors/glass" convention shading.mojo's path.albedo AOV
    already uses, needed for the denoiser's albedo guide buffer (see
    vcm_render's docstring). `use_gpu` now genuinely matters: it selects
    _tex_lookup's CPU (tex_filenames/OIIO) vs GPU (GpuTexture_C array)
    texture-sampling branch for diffuse/coateddiffuse vertex albedo — CPU
    and GPU callers MUST pass the value matching their own reality (the
    CPU driver previously passed [False] here anyway, so this was already
    correct; VCM Stage 2c's texture fix (task #150) is what makes it
    load-bearing) — kept as a parameter so its signature matches
    `_bdpt_trace_light_path`'s and the two thin kernels wrapping this stay
    symmetric in Phase (b).

    `lp_idx`/`path_len` (VCM Stage 2b) identify this pixel's own
    DETERMINISTICALLY PAIRED light path (n_light_paths == n_pix, standard
    Veach BDPT pairing — see _bdpt_store_lvc_vertex's docstring and
    project_vcm_stage2_mis_derivation memory) — `_bdpt_connect_to_cache`
    connects to every one of that path's `path_len` stored vertices, an
    exhaustive sum over its depth-strategies, no random subsampling or
    rescaling needed."""
    var fX = Float32(px) + Float32(0.5)
    var fY = Float32(py) + Float32(0.5)
    var cx = r2c[0]*fX + r2c[4]*fY + r2c[12]
    var cy = r2c[1]*fX + r2c[5]*fY + r2c[13]
    var cz = r2c[2]*fX + r2c[6]*fY + r2c[14]
    var cw = r2c[3]*fX + r2c[7]*fY + r2c[15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = sqrt(cx*cx + cy*cy + cz*cz)
    if cl > Float32(0.0): cx /= cl; cy /= cl; cz /= cl
    var rd = Vec3f(
        c2w[0]*cx + c2w[4]*cy + c2w[8]*cz,
        c2w[1]*cx + c2w[5]*cy + c2w[9]*cz,
        c2w[2]*cx + c2w[6]*cy + c2w[10]*cz,
    )
    var dl = rd.length()
    if dl > Float32(0.0): rd = rd / dl
    var ro = Point3f(c2w[12], c2w[13], c2w[14])

    # VCM Stage 2b: real per-vertex MIS state for the eye subpath (see
    # project_vcm_stage2_mis_derivation memory). cameraPdfW derived by
    # analogy with SmallVCM's pinhole-camera imageToSolidAngleFactor,
    # using px_scale (world-space size of one pixel at unit distance along
    # the camera forward axis — same quantity the plain path tracer's mip
    # LOD already uses, pipeline.mojo) as the "pixel area at distance 1"
    # convention: cameraPdfW = 1/(px_scale² × cosθ³), cosθ = angle between
    # this ray and the camera forward axis. cz here is already normalized
    # (post `cl` division above) so |cz| IS that cosine directly.
    var cos_theta_at_camera = abs(cz)
    var camera_pdf_w = Float32(1) / max(
        px_scale * px_scale * cos_theta_at_camera * cos_theta_at_camera * cos_theta_at_camera,
        Float32(1e-12))
    var dvcm_carry = n_light_paths_f / camera_pdf_w
    var dvc_carry = Float32(0)
    var dvm_carry = Float32(0)

    var n_verts = 0
    var n_bounces = 0  # total surface hits including glass (for _dielectric_bounce entering logic)
    var beta = RGB(Float32(1))
    var cur_med_idx = start_med_idx
    var total = RGB(Float32(0))
    var first_alb = RGB(Float32(0))  # denoiser albedo AOV -- set at the first stored vertex, below
    # One hero-wavelength sample per camera subpath (staged spectral
    # rendering rollout, Stage 3 -- see project_spectral_rendering memory),
    # stored into every vertex this subpath constructs and used for this
    # subpath's own NEE terms.
    var wavelengths = sample_wavelengths_uniform(pcg.next_float())
    # pdf (solid angle) of the cosine-weighted diffuse bounce that produced
    # the CURRENT `rd`, used to MIS-weight this ray's eventual infinite-light
    # miss-escape contribution against the NEE-to-infinite-light sample taken
    # at the vertex that generated it (see the diffuse branch below and the
    # miss handler). -1 = no competing NEE strategy exists for whatever
    # generated this ray (primary ray, or a conductor/dielectric/volume
    # bounce — none of which do infinite-light NEE in this function) → full
    # weight, no MIS needed. Mirrors shading.mojo's lastBsdfPdf bookkeeping.
    var last_bsdf_pdf = Float32(-1)

    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(ro, rd)
        scratch[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        if scratch[0].hit == Int8(0):
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ilight = sd.infiniteLights[inf_i]
                var (Le, pdf_light_here) = _eval_infinite_light_and_pdf(ilight, rd)
                var mis_w = Float32(1)
                if last_bsdf_pdf >= Float32(0) and pdf_light_here > Float32(0):
                    mis_w = power_heuristic(last_bsdf_pdf, pdf_light_here)
                total += beta * Le * mis_w
            break

        var inter = scratch[0]
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()

        # VCM Stage 2b: distance-squared portion of the per-bounce MIS
        # correction -- see _bdpt_trace_light_path's matching comment
        # (project_vcm_stage2_mis_derivation memory). The eye subpath's
        # origin is always "finite" (a real camera position), so the
        # correction applies unconditionally here (unlike the light side's
        # is_finite_origin check).
        dvcm_carry *= t_hit * t_hit

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                # Volume scatter vertex
                var sp = ro + rd*ff.t_free
                # BDPT vertex beta: Tr/pdf_free × phase/pdf_phase = 1/sig_t × sig_s = alb_s
                # (exp(-sig_t×t) cancels between Tr numerator and pdf denominator)
                var v = _null_vertex()
                v.pos = sp
                v.beta = beta * ff.albedo
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.sig_t * exp(-ff.sig_t * ff.t_free)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                if n_verts == 0: first_alb = ff.albedo
                n_verts += 1
                # Volume: out of MIS scope this pass, same as the light side.
                dvcm_carry = Float32(0)
                if path_len > 0:
                    # BUG FIX (found investigating volumetric-caustic's
                    # colored-blob fireflies): merge is only safe to run
                    # unconditionally alongside connect when BOTH already
                    # carry a real per-candidate MIS weight that makes their
                    # SUM correct (see this file's opening VCM comment).
                    # Volume vertices are out of _bdpt_vertex_mis_scoped's
                    # scope, so _bdpt_merge_from_cache's own weight defaults
                    # to 1 -- summing merge(w=1) + connect(w=1) here would
                    # double the correct answer (E[merge]+E[connect] = 2I,
                    # not I), exactly the double-counting the module's
                    # original Stage-1 stochastic-pick design was built to
                    # avoid. Only call merge when this vertex is actually
                    # MIS-scoped; otherwise connect alone (unweighted) is
                    # already the complete, correct estimate -- Stage 2b's
                    # original, verified behavior for out-of-scope kinds.
                    if _bdpt_vertex_mis_scoped(v):
                        total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                    total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)
                # Continuation beta = prev × alb_s (same as stored vertex beta)
                beta *= ff.albedo
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                last_bsdf_pdf = Float32(-1)  # isotropic phase scatter: no infinite-light NEE done here
                continue
            else:
                # Beer-Lambert through full segment to surface
                beta *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd*t_hit

        # Direct hit on an emissive analytic sphere — checked BEFORE material
        # dispatch since the sphere's own material is often an inert
        # placeholder (e.g. pbrt's "Null" material on AreaLightSource
        # spheres), so mat.type never reflects that this primitive emits;
        # Sphere_C.isAreaLight is the only way to know. MIS-weighted against
        # last_bsdf_pdf — same bookkeeping already used for infinite-light
        # miss-escape, since a competing NEE strategy toward this SAME
        # sphere may have already been taken at the PREVIOUS vertex (see
        # this function's own sphere-light NEE blocks). No facing check —
        # an analytic sphere is only ever hit from outside, always the
        # front (emitting) face, unlike a one-sided triangle area light.
        if inter.primId.type == Int8(4):
            var sph_hit = sd.spheres[Int(inter.primId.id1)]
            if sph_hit.isAreaLight != Int8(0):
                var mis_w_sph_hit = Float32(1)
                if last_bsdf_pdf >= Float32(0):
                    var to_c_hit = sph_hit.center - hit
                    var dc_sq_hit = to_c_hit.length_sq()
                    var sin2_max_hit = sph_hit.radius * sph_hit.radius / dc_sq_hit
                    if sin2_max_hit < Float32(1):
                        var cos_max_hit = sqrt(Float32(1) - sin2_max_hit)
                        var solid_angle_hit = Float32(2) * PI * (Float32(1) - cos_max_hit)
                        var n_sph_hit = Float32(max(Int(sd.sphereCount), 1))
                        var pdf_light_hit = Float32(1) / (solid_angle_hit * n_sph_hit)
                        mis_w_sph_hit = power_heuristic(last_bsdf_pdf, pdf_light_hit)
                total += beta * sph_hit.emission * mis_w_sph_hit
                break

        # Mix material: stochastically resolve to one of two sub-materials
        # (mirrors shading.mojo's shade_mix) before any type dispatch below —
        # packing/guard-against-mix-of-mix convention identical to shade_mix.
        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[mix_chosen]
            mat_idx = mix_chosen  # keep in sync with the resolved sub-material (hair needs the real index to re-fetch at connect time)
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

        if mat.type == MatKind.area_light:
            # Direct hit on a triangle/curve area light — same MIS-against-
            # last_bsdf_pdf treatment as the sphere case above. id1 is the
            # AreaLight_C index directly for a type==3 (area-light-triangle)
            # hit, per pbrt_parser.mojo's own PrimId_C encoding.
            var al_hit = sd.areaLights[Int(inter.primId.id1)]
            var gn_al_hit = _geom_normal(inter, sd.meshes, sd.instances)
            var cos_l_hit = -dot(gn_al_hit, ray_dir)
            if cos_l_hit > Float32(0):
                var mis_w_al_hit = Float32(1)
                if last_bsdf_pdf >= Float32(0):
                    var dist2_hit = t_hit * t_hit
                    var n_area_hit = Float32(max(Int(sd.areaLightCount), 1))
                    var pdf_light_al = dist2_hit / (cos_l_hit * n_area_hit * al_hit.total_area)
                    mis_w_al_hit = power_heuristic(last_bsdf_pdf, pdf_light_al)
                total += beta * al_hit.emission * mis_w_al_hit
            break

        elif mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            # VCM Stage 2b: finish the per-bounce MIS correction (dist²
            # portion already applied above) -- see
            # _bdpt_trace_light_path's matching comment.
            var cos_fix = abs(dot(-ray_dir, gn))
            if cos_fix > Float32(1e-6):
                dvcm_carry /= cos_fix
                dvc_carry /= cos_fix
                dvm_carry /= cos_fix
            # Real image-texture reflectance (e.g. "texture reflectance" on
            # coateddiffuse) — before this, bdpt.mojo always used the flat
            # mat.albedo fallback (material_builder.mojo's own 0.5 grey
            # default for any texture-backed material), silently washing
            # out any textured diffuse/coateddiffuse surface. _tex_lookup
            # itself returns mat.albedo unchanged when mat.tex_idx == -1
            # (no texture), so this is a strict improvement, never a
            # regression, for flat-color materials.
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn)
            v.beta = beta
            v.alb = eff_alb
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1)  # unused by the uniform-subsample estimator
            v.wo = vec3f(-ray_dir)  # VCM Stage 2b: needed for _connect's reverse-pdf eval
            v.med_idx = cur_med_idx
            v.wavelengths = wavelengths
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            if n_verts == 0: first_alb = eff_alb
            n_verts += 1
            if path_len > 0:
                total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)

            # NEE to distant/point/sphere/infinite lights, via the shared
            # Light interface (bvh.mojo's LightSample samplers) + BxDF
            # interface (bxdf.mojo's _nee_weight_simple) + this file's own
            # _bdpt_nee_contribute glue (media-aware transmittance test) —
            # replacing 4 formerly hand-inlined blocks also duplicated in
            # the conductor branch below and in _bdpt_trace_light_path. Area
            # lights are NOT covered: the LVC cache-connection strategy
            # above already handles them — see _bdpt_trace_light_path's
            # docstring for why distant/infinite/point/sphere need this
            # separate direct term instead.
            var wo_d = -ray_dir
            for dl_i in range(Int(sd.distantLightCount)):
                var ls_d = _sample_distant_light_nee(sd.distantLights[dl_i])
                var w_d = _nee_weight_simple_via_spectral(ls_d, Int32(0), eff_alb, Float32(0), gn, wo_d, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                total += _bdpt_nee_contribute(beta, w_d, ls_d, hit, gn, cur_med_idx, sd, scratch)
            for pl_i in range(Int(sd.pointLightCount)):
                var ls_p = _sample_point_light_nee(sd.pointLights[pl_i], hit.to_simd())
                var w_p = _nee_weight_simple_via_spectral(ls_p, Int32(0), eff_alb, Float32(0), gn, wo_d, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                total += _bdpt_nee_contribute(beta, w_p, ls_p, hit, gn, cur_med_idx, sd, scratch)
            for sph_i in range(Int(sd.sphereCount)):
                var ls_sph = _sample_sphere_light_nee(sd.spheres[sph_i], Int(sd.sphereCount), hit.to_simd(), pcg)
                var w_sph = _nee_weight_simple_via_spectral(ls_sph, Int32(0), eff_alb, Float32(0), gn, wo_d, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                total += _bdpt_nee_contribute(beta, w_sph, ls_sph, hit, gn, cur_med_idx, sd, scratch)
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ls_e = _sample_infinite_light_nee(sd.infiniteLights[inf_i], Point2f(pcg.next_float(), pcg.next_float()))
                var w_e = _nee_weight_simple_via_spectral(ls_e, Int32(0), eff_alb, Float32(0), gn, wo_d, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                total += _bdpt_nee_contribute(beta, w_e, ls_e, hit, gn, cur_med_idx, sd, scratch)

            # Cosine-weighted scatter direction
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            rd = vec3f(_cosine_hemisphere_sample(gn, u1, u2))
            ro = hit + rd*Float32(0.0002)
            last_bsdf_pdf = bxdf_pdf_diffuse(dot(gn, rd.to_simd()))
            # Update beta: f/pdf for Lambertian = (alb/π) / (cosθ/π) = alb
            beta *= eff_alb
            # VCM Stage 2b: recursive continuation for the NEXT bounce --
            # see _bdpt_trace_light_path's matching diffuse-branch comment.
            var cos_theta_out = abs(dot(rd.to_simd(), gn))
            var bsdf_rev_pdf_w = cos_fix / PI
            var dvc_new = PI * (dvc_carry * bsdf_rev_pdf_w + dvcm_carry + mis_vm_weight_factor)
            var dvm_new = PI * (dvm_carry * bsdf_rev_pdf_w + dvcm_carry * mis_vc_weight_factor + Float32(1))
            var bsdf_dir_pdf_w = cos_theta_out / PI
            dvcm_carry = Float32(1) / bsdf_dir_pdf_w if bsdf_dir_pdf_w > Float32(1e-8) else Float32(0)
            dvc_carry = dvc_new
            dvm_carry = dvm_new

        elif mat.type == MatKind.conductor or mat.type == MatKind.coated_conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                gn_c = sphere_outward_normal(hit, sph_c.center).to_simd()
            else:
                gn_c = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=SIMD[DType.float32, 3](frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=SIMD[DType.float32, 3](frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=mat.albedo, pixel_uv=Float32(0),
            )
            var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
            var bs_c: BxDFSample
            if mat.type == MatKind.conductor:
                bs_c = bxdf_sample_conductor(gc_c, mat, uc1, uc2)
            else:
                # Coated conductor: dielectric clearcoat over GGX conductor —
                # bxdf_sample_coated_conductor's own u_split picks coat-vs-
                # conductor lobe; its delta (coat-reflect) branch is treated
                # exactly like mirror conductor/dielectric elsewhere in this
                # function (no stored vertex), its glossy (conductor) branch
                # exactly like plain conductor (approximation: connections
                # reuse conductor's own GGX eval/f0, ignoring the coat's own
                # (1-f_coat) attenuation and its separate luma-Fresnel blend
                # — same approximation shading.mojo's sampling side already
                # makes for this material).
                var ior_c = mat.emission.r if mat.emission.r > Float32(1) else Float32(1.5)
                var usplit_c = pcg.next_float()
                bs_c = bxdf_sample_coated_conductor(gc_c, mat, ior_c, usplit_c, uc1, uc2)
            if bs_c.is_valid == Int8(0):
                break
            var alpha_c = max(mat.roughU, mat.roughV)
            # VCM Stage 2d: rough conductor DOES have a real standalone pdf
            # (bxdf_pdf_conductor_ggx, the same Heitz 2018 VNDF density
            # bxdf_sample_conductor's glossy branch itself samples from) --
            # finish the per-bounce MIS correction the same way diffuse
            # does (dist² portion already applied above).
            var cos_fix_c = abs(dot(-ray_dir, gn_c))
            if cos_fix_c > Float32(1e-6):
                dvc_carry /= cos_fix_c
                dvm_carry /= cos_fix_c
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c)
                v.beta = beta
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = alpha_c
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                if n_verts == 0: first_alb = mat.albedo
                n_verts += 1
                if path_len > 0:
                    total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                    total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)

                # Distant/point/sphere/infinite NEE, via the shared Light
                # interface + BxDF interface + _bdpt_nee_contribute glue —
                # replacing 4 formerly hand-inlined blocks also duplicated
                # in the diffuse branch above and _bdpt_trace_light_path.
                # NOTE: the old inline sphere-light block here was missing
                # its cosine factor (computed shadow_dir but never took
                # dot(gn_c, shadow_dir) before dividing by pdf) — a real
                # overbrightness bug, fixed as a side effect of routing
                # through the shared, already-correct _nee_weight_simple.
                for dl_ic in range(Int(sd.distantLightCount)):
                    var ls_dc = _sample_distant_light_nee(sd.distantLights[dl_ic])
                    var w_dc = _nee_weight_simple_via_spectral(ls_dc, Int32(1), mat.albedo, alpha_c, gn_c, wo_c, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                    total += _bdpt_nee_contribute(beta, w_dc, ls_dc, hit, gn_c, cur_med_idx, sd, scratch)
                for pl_ic in range(Int(sd.pointLightCount)):
                    var ls_pc = _sample_point_light_nee(sd.pointLights[pl_ic], hit.to_simd())
                    var w_pc = _nee_weight_simple_via_spectral(ls_pc, Int32(1), mat.albedo, alpha_c, gn_c, wo_c, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                    total += _bdpt_nee_contribute(beta, w_pc, ls_pc, hit, gn_c, cur_med_idx, sd, scratch)
                for sph_ic in range(Int(sd.sphereCount)):
                    var ls_sphc = _sample_sphere_light_nee(sd.spheres[sph_ic], Int(sd.sphereCount), hit.to_simd(), pcg)
                    var w_sphc = _nee_weight_simple_via_spectral(ls_sphc, Int32(1), mat.albedo, alpha_c, gn_c, wo_c, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                    total += _bdpt_nee_contribute(beta, w_sphc, ls_sphc, hit, gn_c, cur_med_idx, sd, scratch)
                for inf_ic in range(Int(sd.infiniteLightCount)):
                    var ls_ec = _sample_infinite_light_nee(sd.infiniteLights[inf_ic], Point2f(pcg.next_float(), pcg.next_float()))
                    var w_ec = _nee_weight_simple_via_spectral(ls_ec, Int32(1), mat.albedo, alpha_c, gn_c, wo_c, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                    total += _bdpt_nee_contribute(beta, w_ec, ls_ec, hit, gn_c, cur_med_idx, sd, scratch)

            beta *= bs_c.f
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)
            var cos_theta_out_c = abs(dot(bs_c.wi, gn_c))
            if bxdf_is_delta(bs_c.flags):
                last_bsdf_pdf = Float32(-1)  # mirror bounce: no infinite-light NEE done at this vertex
                # VCM: specular bounce -- same dVCM=0/cosine-rescale reset
                # SmallVCM's own delta-bounce case uses (vertexcm.hxx's
                # SampleScattering, kSpecular branch); matches this file's
                # diffuse/dielectric specular handling elsewhere.
                dvcm_carry = Float32(0)
                dvc_carry *= cos_theta_out_c
                dvm_carry *= cos_theta_out_c
            else:
                # VCM Stage 2d: real non-specular recursive update using the
                # actual VNDF sampling density -- see
                # _bdpt_trace_light_path's matching comment for the general
                # (cosThetaOut/bsdfDirPdfW) form (doesn't simplify to a
                # constant like diffuse's PI does, since GGX's pdf isn't
                # proportional to cos_theta_out).
                var bsdf_dir_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, wo_c, bs_c.wi, alpha_c)
                last_bsdf_pdf = bsdf_dir_pdf_w_c
                if bsdf_dir_pdf_w_c > Float32(1e-8):
                    var bsdf_rev_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, bs_c.wi, wo_c, alpha_c)
                    var inv_pdf_c = cos_theta_out_c / bsdf_dir_pdf_w_c
                    var dvc_new_c = inv_pdf_c * (dvc_carry * bsdf_rev_pdf_w_c + dvcm_carry + mis_vm_weight_factor)
                    var dvm_new_c = inv_pdf_c * (dvm_carry * bsdf_rev_pdf_w_c + dvcm_carry * mis_vc_weight_factor + Float32(1))
                    dvcm_carry = Float32(1) / bsdf_dir_pdf_w_c
                    dvc_carry = dvc_new_c
                    dvm_carry = dvm_new_c
                else:
                    dvcm_carry = Float32(0)
                    dvc_carry = Float32(0)
                    dvm_carry = Float32(0)

        elif mat.type == MatKind.hair:
            # Marschner 3-lobe hair BSDF — no delta lobe, so always store a
            # connectible vertex (unlike conductor's mirror-vs-rough split)
            # and always importance-sample a continuation direction, mirroring
            # shading.mojo's own shade_hair (via the shared bvh.mojo helpers).
            var curve_idx_h = Int(inter.primId.id1)
            var wo_h = (-rd).to_simd()
            var hc = _hair_precompute(mat, sd.curves, curve_idx_h, inter.v, inter.u, wo_h)
            var hair_eps = curve_offset_eps(hc.radius)
            # VCM Stage 2b: hair has no real standalone pdf either -- same
            # out-of-scope treatment as conductor.
            var cos_fix_h = abs(dot(-ray_dir, hc.geo_normal))
            if cos_fix_h > Float32(1e-6):
                dvc_carry /= cos_fix_h
                dvm_carry /= cos_fix_h
            var v_h = _null_vertex()
            v_h.pos = hit
            v_h.normal = vec3f(hc.geo_normal)
            v_h.beta = beta
            v_h.alb = mat.albedo
            v_h.is_surface = Int32(1); v_h.is_delta = Int32(0); v_h.mat_kind = Int32(2)
            v_h.wo = vec3f(wo_h)
            v_h.mat_idx = Int32(mat_idx)
            v_h.hair_curve_idx = Int32(curve_idx_h)
            v_h.hair_h = inter.u
            v_h.hair_v = inter.v
            v_h.pdf_fwd = Float32(1)
            v_h.med_idx = cur_med_idx
            v_h.wavelengths = wavelengths
            if n_verts == 0: first_alb = mat.albedo
            n_verts += 1
            if path_len > 0:
                # Hair is out of MIS scope -- see the volume branch's
                # matching comment above for why merge must be skipped
                # (not just unweighted) when connect already fires
                # unconditionally, to avoid double-counting.
                if _bdpt_vertex_mis_scoped(v_h):
                    total += _bdpt_merge_from_cache(v_h, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                total += _bdpt_connect_to_cache(v_h, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)

            # Distant/point/sphere/infinite NEE, via the shared Light
            # interface + BxDF interface (_nee_weight_hair, using `hc`) +
            # _bdpt_nee_contribute glue — replacing 3 formerly hand-inlined
            # blocks also duplicated in shading.mojo's shade_hair; hair has
            # no delta lobe, so this always applies. Shadow-ray origin stays
            # a fixed +geo_normal offset (no sign-flip toward wi, unlike
            # shading.mojo's shade_hair) — preserves this file's own
            # existing convention. Sphere-light NEE is new (this branch
            # previously had none).
            for dl_ih in range(Int(sd.distantLightCount)):
                var ls_dh = _sample_distant_light_nee(sd.distantLights[dl_ih])
                var w_dh = _nee_weight_hair(ls_dh, hc)
                total += _bdpt_nee_contribute(beta, w_dh, ls_dh, hit, hc.geo_normal, cur_med_idx, sd, scratch, hair_eps)
            for pl_ih in range(Int(sd.pointLightCount)):
                var ls_ph = _sample_point_light_nee(sd.pointLights[pl_ih], hit.to_simd())
                var w_ph = _nee_weight_hair(ls_ph, hc)
                total += _bdpt_nee_contribute(beta, w_ph, ls_ph, hit, hc.geo_normal, cur_med_idx, sd, scratch, hair_eps)
            for sph_ih in range(Int(sd.sphereCount)):
                var ls_sphh = _sample_sphere_light_nee(sd.spheres[sph_ih], Int(sd.sphereCount), hit.to_simd(), pcg)
                var w_sphh = _nee_weight_hair(ls_sphh, hc)
                total += _bdpt_nee_contribute(beta, w_sphh, ls_sphh, hit, hc.geo_normal, cur_med_idx, sd, scratch, hair_eps)
            for inf_ih in range(Int(sd.infiniteLightCount)):
                var ls_eh = _sample_infinite_light_nee(sd.infiniteLights[inf_ih], Point2f(pcg.next_float(), pcg.next_float()))
                var w_eh = _nee_weight_hair(ls_eh, hc)
                total += _bdpt_nee_contribute(beta, w_eh, ls_eh, hit, hc.geo_normal, cur_med_idx, sd, scratch, hair_eps)

            var (wi_hs, f_hs, pdf_hs, cos_ti_hs2) = _hair_sample_dir(hc, pcg)
            beta *= f_hs / pdf_hs
            rd = vec3f(wi_hs)
            var hsign = Float32(1) if dot(wi_hs, hc.geo_normal) >= Float32(0) else Float32(-1)
            ro = hit + vec3f(hc.geo_normal) * hair_eps * hsign
            last_bsdf_pdf = pdf_hs * cos_ti_hs2  # solid-angle pdf (pdf_hs already has /cos_ti baked in)
            var cos_theta_out_h = abs(dot(wi_hs, hc.geo_normal))
            dvcm_carry = Float32(0)
            dvc_carry *= cos_theta_out_h
            dvm_carry *= cos_theta_out_h

        elif mat.type == MatKind.measured:
            # Tabulated Dupuy & Jakob MeasuredBxDF -- the real algorithm
            # (measured_bxdf_eval.mojo), not an approximation, via the same
            # shared bxdf.mojo/measured_bxdf_eval.mojo interface
            # shading.mojo's shade_measured already uses. Isotropic only (see
            # the loader's scope note), so an arbitrary Frisvad tangent frame
            # is fine -- no UV alignment needed, same reasoning as conductor.
            var gn_m: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_m = Int(inter.primId.id1)
                var sph_m = sd.spheres[si_m]
                gn_m = sphere_outward_normal(hit, sph_m.center).to_simd()
            else:
                gn_m = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_m, ray_dir) > Float32(0): gn_m = gn_m * Float32(-1)
            if mat.measured_idx < Int32(0):
                # Load failure fallback (see material_builder.mojo) -- matches
                # shading.mojo's shade_measured: stop this path rather than
                # dereferencing a nonexistent measuredBrdfs entry.
                break
            var wo_m = (-rd).to_simd()
            var frm_m = Frame.from_z(Vec3f(gn_m[0], gn_m[1], gn_m[2]))
            var tangent_m = SIMD[DType.float32, 3](frm_m.x.x, frm_m.x.y, frm_m.x.z)
            var bitangent_m = SIMD[DType.float32, 3](frm_m.y.x, frm_m.y.y, frm_m.y.z)
            var mb = sd.measuredBrdfs[Int(mat.measured_idx)]

            # VCM Stage 2b: measured BxDF has a real standalone pdf -- in
            # MIS scope this pass, same real treatment as diffuse.
            var cos_fix_m = abs(dot(-ray_dir, gn_m))
            if cos_fix_m > Float32(1e-6):
                dvcm_carry /= cos_fix_m
                dvc_carry /= cos_fix_m
                dvm_carry /= cos_fix_m

            var v_m = _null_vertex()
            v_m.pos = hit
            v_m.normal = vec3f(gn_m)
            v_m.beta = beta
            v_m.alb = mat.albedo
            v_m.is_surface = Int32(1); v_m.is_delta = Int32(0); v_m.mat_kind = Int32(3)
            v_m.wo = vec3f(wo_m)
            v_m.mat_idx = Int32(mat_idx)
            v_m.pdf_fwd = Float32(1)
            v_m.med_idx = cur_med_idx
            v_m.wavelengths = wavelengths
            v_m.dVCM = dvcm_carry; v_m.dVC = dvc_carry; v_m.dVM = dvm_carry
            if n_verts == 0: first_alb = mat.albedo
            n_verts += 1
            if path_len > 0:
                # Measured is now IN MIS scope (task #153 closed the
                # connect/merge-time reverse-pdf local-frame gap -- see
                # _bdpt_vertex_pdfs' mat_kind=3 branch), so this check is
                # always True here; kept for symmetry with the volume/hair
                # branches' matching guard against merge+connect double-
                # counting for any future out-of-scope kind.
                if _bdpt_vertex_mis_scoped(v_m):
                    total += _bdpt_merge_from_cache(v_m, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                total += _bdpt_connect_to_cache(v_m, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)

            # Distant/point/sphere/infinite NEE, via the shared Light
            # interface + BxDF interface (_nee_weight_measured) +
            # _bdpt_nee_contribute glue -- same pattern as every other
            # material branch above.
            for dl_im in range(Int(sd.distantLightCount)):
                var ls_dm = _sample_distant_light_nee(sd.distantLights[dl_im])
                var w_dm = _nee_weight_measured(ls_dm, mb, tangent_m, bitangent_m, gn_m, wo_m, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                total += _bdpt_nee_contribute(beta, w_dm, ls_dm, hit, gn_m, cur_med_idx, sd, scratch)
            for pl_im in range(Int(sd.pointLightCount)):
                var ls_pm = _sample_point_light_nee(sd.pointLights[pl_im], hit.to_simd())
                var w_pm = _nee_weight_measured(ls_pm, mb, tangent_m, bitangent_m, gn_m, wo_m, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                total += _bdpt_nee_contribute(beta, w_pm, ls_pm, hit, gn_m, cur_med_idx, sd, scratch)
            for sph_im in range(Int(sd.sphereCount)):
                var ls_sm = _sample_sphere_light_nee(sd.spheres[sph_im], Int(sd.sphereCount), hit.to_simd(), pcg)
                var w_sm = _nee_weight_measured(ls_sm, mb, tangent_m, bitangent_m, gn_m, wo_m, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                total += _bdpt_nee_contribute(beta, w_sm, ls_sm, hit, gn_m, cur_med_idx, sd, scratch)
            for inf_im in range(Int(sd.infiniteLightCount)):
                var ls_em = _sample_infinite_light_nee(sd.infiniteLights[inf_im], Point2f(pcg.next_float(), pcg.next_float()))
                var w_em = _nee_weight_measured(ls_em, mb, tangent_m, bitangent_m, gn_m, wo_m, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                total += _bdpt_nee_contribute(beta, w_em, ls_em, hit, gn_m, cur_med_idx, sd, scratch)

            var wo_l_m = SIMD[DType.float32, 3](dot(wo_m, tangent_m), dot(wo_m, bitangent_m), dot(wo_m, gn_m))
            var um1 = pcg.next_float(); var um2 = pcg.next_float()
            var (wi_l_m, f_m, pdf_m, valid_m) = bxdf_sample_measured(mb, wo_l_m, um1, um2, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
            if not valid_m or pdf_m <= Float32(0):
                break
            var wi_m = tangent_m * wi_l_m[0] + bitangent_m * wi_l_m[1] + gn_m * wi_l_m[2]
            var wilen_m = dot(wi_m, wi_m)
            if wilen_m > Float32(0):
                wi_m = wi_m * (Float32(1.0) / sqrt(wilen_m))
            var cos_wi_m = dot(wi_m, gn_m)
            if cos_wi_m <= Float32(0):
                break
            beta *= f_m * (cos_wi_m / pdf_m)
            rd = vec3f(wi_m)
            ro = hit + rd*Float32(0.0002)
            last_bsdf_pdf = pdf_m
            # VCM Stage 2b: recursive continuation, real forward/reverse pdf
            # from bxdf_pdf_measured -- see _bdpt_trace_light_path's
            # matching measured-branch comment (reverse-pdf convention
            # ASSUMED, not independently verified).
            var pdf_rev_m = bxdf_pdf_measured(mb, wi_l_m, wo_l_m)
            var dvc_new_m = (cos_wi_m / pdf_m) * (dvc_carry * pdf_rev_m + dvcm_carry + mis_vm_weight_factor)
            var dvm_new_m = (cos_wi_m / pdf_m) * (dvm_carry * pdf_rev_m + dvcm_carry * mis_vc_weight_factor + Float32(1))
            dvcm_carry = Float32(1) / pdf_m if pdf_m > Float32(1e-8) else Float32(0)
            dvc_carry = dvc_new_m
            dvm_carry = dvm_new_m

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var (new_dir, new_org, radiance_scale) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, mat.albedo.r, n_bounces, pcg)
            n_bounces += 1
            last_bsdf_pdf = Float32(-1)  # delta bounce: no infinite-light NEE done here
            # Specular vertex: no BSDF record needed, just track throughput.
            # Camera path (Radiance mode): apply the non-symmetric-scattering
            # correction (see _dielectric_bounce's docstring).
            beta *= radiance_scale
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rd = vec3f(new_dir)
            ro = point3f(new_org)
            # VCM Stage 2b: genuinely delta/specular, matches SmallVCM's own
            # specular-bounce handling directly.
            var cos_fix_d = abs(dot(-ray_dir, gn))
            if cos_fix_d > Float32(1e-6):
                dvc_carry /= cos_fix_d
                dvm_carry /= cos_fix_d
            var cos_theta_out_d = abs(dot(new_dir, gn))
            dvcm_carry = Float32(0)
            dvc_carry *= cos_theta_out_d
            dvm_carry *= cos_theta_out_d

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)
            # VCM Stage 2b: pure pass-through, carry unchanged (see
            # _bdpt_trace_light_path's matching interface-branch comment).

        else:
            break

    return (total, first_alb)

# ── Trace one light subpath, storing its vertices into the shared cache ─────

def _bdpt_trace_light_path[use_gpu: Bool](
    sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    default_emit_med: Int32,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc:      UnsafePointer[BDPTVertex, MutAnyOrigin],
    lp_idx:   Int,
    lvc_path_len: UnsafePointer[Int32, MutAnyOrigin],
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
):
    """Emit a photon from a random light and trace a light subpath, storing
    every non-delta vertex (including the light-source point itself, the
    s=1/NEE-equivalent strategy) into light path `lp_idx`'s own dedicated
    LVC slice via `_bdpt_store_lvc_vertex` — see the module's LVC-BPT
    docstring above and that function's docstring for the VCM Stage 2b
    per-path-indexed storage layout. `_BDPT_MAX_VERTS` caps how many
    vertices any ONE light path contributes; `lvc_path_len[lp_idx]` records
    how many of those slots this path actually filled (may be fewer than
    _BDPT_MAX_VERTS if the path terminated early), initialized to 0 up
    front so every exit path (including the early returns below) leaves it
    correctly set.

    `lp_idx` is also the pixel index this light path is deterministically
    PAIRED with for connections (n_light_paths == n_pix, one light path per
    pixel) — real VCM's dVCM/dVC/dVM MIS weights assume this pairing (see
    project_vcm_stage2_mis_derivation memory), unlike the old design's
    random draws from a shared pool across all pixels.

    Lights are chosen uniformly across ALL light types (area + distant +
    infinite), not just area lights — `n_lights` below is this combined
    total, so every per-light PDF-correction factor (`area_weight`, the
    distant/infinite flux scale) uses it, not just an area-only count.
    Distant/infinite lights have no finite position of their own, so unlike
    the area-light case, no vertex-0 (`lv0_vert`) is stored for them — a
    disk-sampled point on the scene's bounding sphere (see bvh.mojo's
    `_scene_bounding_sphere`/`_sample_disk_perpendicular`) isn't a real
    scene point and its G(a,b) inverse-square/cosine term would be
    physically wrong for a directional source. Direct (NEE-equivalent)
    illumination from these lights is instead provided by
    `_bdpt_trace_camera_and_connect`'s own per-vertex NEE to distant/
    infinite lights — this function only needs to seed a physically correct
    ray+flux and let the ordinary bounce loop below take over once that ray
    hits a real surface (which DOES get stored/connected normally)."""
    var n_area = Int(sd.areaLightCount)
    var n_distant = Int(sd.distantLightCount)
    var n_infinite = Int(sd.infiniteLightCount)
    var n_point = Int(sd.pointLightCount)
    var n_lights = n_area + n_distant + n_infinite + n_point
    lvc_path_len[lp_idx] = Int32(0)
    if n_lights == 0:
        return

    var ro: Point3f
    var rd: Vec3f
    var flux: RGB
    var n_verts: Int
    # VCM Stage 2b: real per-vertex MIS state, recursively carried along
    # this light subpath (see project_vcm_stage2_mis_derivation memory for
    # the verified formulas). Only AREA lights get a real, non-zero origin
    # this pass — distant/infinite/point-seeded paths start at 0, a
    # deliberately scoped simplification (their own MIS treatment needs
    # the delta-light formulas Georgiev's tech report's (48)-(50) define,
    # not yet ported). Only diffuse/measured vertices update these
    # meaningfully; every other material resets them as if specular
    # (dVCM=0, dVC/dVM *= cosThetaOut) since they don't have a real
    # separate forward/reverse pdf today — see the memory file.
    var dvcm_carry = Float32(0)
    var dvc_carry = Float32(0)
    var dvm_carry = Float32(0)
    var is_finite_origin = False
    # One hero-wavelength sample per light subpath (see
    # _bdpt_trace_camera_and_connect's matching comment) — this subpath's own
    # NEE is done from the CAMERA side (this function's own docstring above),
    # so `wavelengths` here is only stored into cache vertices, used later at
    # whichever camera vertex's own wavelength-consistent connection picks
    # this one from the LVC.
    var wavelengths = sample_wavelengths_uniform(pcg.next_float())

    var light_pick = Int(pcg.next_uint() % UInt32(n_lights))
    if light_pick < n_area:
        # Pick a light uniformly + a random triangle + barycentric point on it.
        var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_area, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = light_sample.normal

        # Cosine-weighted emission direction
        var du1 = pcg.next_float(); var du2 = pcg.next_float()
        var pdir = _cosine_hemisphere_sample(ln, du1, du2)

        # Light vertex (the emission point, s=1 strategy connects here directly).
        # beta = 1/p_A = area × n_lights (area sampling PDF correction only).
        # alb  = Le (emission) — used as f_lgt in _connect for the light vertex.
        # G already handles the cos_l factor, so f_lgt = Le (no extra cos multiply).
        var area_weight = al.total_area * Float32(n_lights)
        var lv0_vert = _null_vertex()
        lv0_vert.pos = point3f(lp)
        lv0_vert.normal = vec3f(ln)
        lv0_vert.beta = RGB(area_weight, area_weight, area_weight)
        lv0_vert.alb = al.emission
        lv0_vert.is_surface = Int32(1); lv0_vert.is_light = Int32(1)
        lv0_vert.pdf_fwd = Float32(1) / area_weight
        lv0_vert.med_idx = default_emit_med
        lv0_vert.wavelengths = wavelengths

        # VCM Stage 2b: real MIS origin state for this (finite, area) light
        # -- see project_vcm_stage2_mis_derivation memory for the verified
        # derivation. cos_theta_emit cancels out of the flux formula above
        # (Malley's method) but is needed again here, unrelated to flux.
        is_finite_origin = True
        var cos_theta_emit = max(dot(pdir, ln), Float32(0.0001))
        var direct_pdf_a = Float32(1) / area_weight
        var emission_pdf_w = direct_pdf_a * cos_theta_emit / PI
        dvcm_carry = direct_pdf_a / emission_pdf_w
        dvc_carry = cos_theta_emit / emission_pdf_w
        # SmallVCM vertexcm.hxx:856 -- light-origin dVM uses mMisVcWeightFactor,
        # NOT mMisVmWeightFactor (verified against the reference source; a
        # variable-name mix-up here was silently inert while merge stayed
        # disabled -- lv.dVM was never read by the connect-only weight, only
        # by RangeQuery::Process's real merge weight, added below).
        dvm_carry = dvc_carry * mis_vc_weight_factor
        lv0_vert.dVCM = dvcm_carry
        lv0_vert.dVC = dvc_carry
        lv0_vert.dVM = dvm_carry
        _bdpt_store_lvc_vertex(lv0_vert, lvc, lp_idx, 0)

        # For traced vertices: beta = Le × cos_θ / (p_A × p_ω) where p_ω = cos_θ/π
        # for cosine-weighted emission -- the cos_θ_emitted terms CANCEL exactly
        # (Malley's method: this is the whole point of cosine-weighted emission
        # sampling, see sppm.mojo's photon-emission flux for the same, correctly
        # cos_θ-free formula). β = Le × area × n_lights × π, no cos_θ term.
        # BUG FIX (2026-07-10): this previously divided by cos_θ_emitted instead
        # of letting it cancel, inflating flux by up to 100x on near-grazing
        # emission directions (clamped at cos_θ=0.01) -- root cause of BDPT's
        # pre-existing indirect-bounce fireflies, see project_bdpt_todo memory.
        flux = al.emission * (area_weight * PI)
        ro = point3f(lp) + vec3f(ln)*Float32(0.0001)
        rd = vec3f(pdir)
        n_verts = 1  # vertex 0 is the light point itself
    elif light_pick < n_area + n_distant:
        var dl = sd.distantLights[light_pick - n_area]
        var (center, radius) = _scene_bounding_sphere(sd)
        var dir = Vec3f(dl.direction.x, dl.direction.y, dl.direction.z)
        var disk_pt = _sample_disk_perpendicular(dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        # Phi_light = emission(irradiance) × disk_area; p_i = 1/n_lights;
        # pdf_pos = 1/disk_area; pdf_dir = 1 (delta) → flux = emission × disk_area × n_lights.
        flux = dl.emission * (Float32(n_lights) * PI * radius * radius)
        ro = disk_pt
        rd = dir
        n_verts = 0  # no finite light point to store as a cache vertex
    elif light_pick < n_area + n_distant + n_infinite:
        var il = sd.infiniteLights[light_pick - n_area - n_distant]
        var (center, radius) = _scene_bounding_sphere(sd)
        # _sample_infinite_light_dir returns env_dir in the NEE convention
        # ("direction FROM a shading point TOWARD the light" — same as
        # shading.mojo's _nee_infinite_light usage). A photon leaving the
        # light travels the opposite way, arriving FROM that direction
        # INTO the scene — negate it for the emitted ray/disk placement.
        var (env_dir, env_rgb, pdf_dir) = _sample_infinite_light_dir(il, Point2f(pcg.next_float(), pcg.next_float()))
        if pdf_dir <= Float32(0):
            return
        var emit_dir = -env_dir
        var disk_pt = _sample_disk_perpendicular(emit_dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        # Same derivation as distant, but pdf_dir is the CDF's solid-angle pdf
        # instead of an implicit delta (=1): flux = radiance × disk_area × n_lights / pdf_dir.
        flux = env_rgb * (Float32(n_lights) * PI * radius * radius / pdf_dir)
        ro = disk_pt
        rd = emit_dir
        n_verts = 0
    else:
        # Point light: a real finite position (unlike distant/infinite), but
        # still no NEE-equivalent cache vertex — see this function's own
        # docstring for why (same reasoning as distant/infinite: direct
        # illumination comes from _bdpt_trace_camera_and_connect's own
        # per-vertex point-light NEE instead). Emits uniformly over the
        # sphere (isotropic point light); pdf_dir = 1/(4π), so
        # flux = intensity × 4π × n_lights (pdf_dir cancels).
        var pll = sd.pointLights[light_pick - n_area - n_distant - n_infinite]
        var u1p = pcg.next_float(); var u2p = pcg.next_float()
        var cos_p = Float32(1) - Float32(2) * u1p
        var sin_p = sqrt(max(Float32(0), Float32(1) - cos_p * cos_p))
        var phi_p = Float32(2) * PI * u2p
        var pdir_p = Vec3f(sin_p * cos(phi_p), sin_p * sin(phi_p), cos_p)
        flux = pll.intensity * (Float32(4) * PI * Float32(n_lights))
        ro = pll.position
        rd = pdir_p
        n_verts = 0

    var cur_med_idx = default_emit_med
    var n_lbounces = 1  # counts all surface hits (1 = not-the-primary-ray, matches area-light convention — see _dielectric_bounce's bounce==0 special case)

    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(ro, rd)
        scratch[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        if scratch[0].hit == Int8(0): break

        var inter = scratch[0]
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()

        # VCM Stage 2b: distance-squared portion of the per-bounce MIS
        # correction (project_vcm_stage2_mis_derivation memory) -- shared
        # across every material branch below, since it only depends on the
        # travel distance, not the hit material. The |cosThetaFix| portion
        # is applied separately inside each branch once its own local
        # normal is known.
        if n_verts >= 1 or is_finite_origin:
            dvcm_carry *= t_hit * t_hit

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                var sp = ro + rd*ff.t_free
                # BDPT vertex beta: exp(-sig_t×t)/pdf_free × alb_s = 1/sig_t × alb_s = alb_s (for sig_t=1)
                var v = _null_vertex()
                v.pos = sp
                v.beta = flux * ff.albedo
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.sig_t * exp(-ff.sig_t * ff.t_free)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                n_verts += 1
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                # Volume scatter is isotropic (no surface normal, no
                # cosThetaFix) -- out of MIS scope this pass (like hair/
                # dielectric), reset as if specular so a LATER
                # diffuse/conductor/measured bounce still gets a well-defined carry.
                dvcm_carry = Float32(0)
                # Continuation: flux = prev × alb_s (same as stored vertex beta)
                flux *= ff.albedo
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                continue
            else:
                flux *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd*t_hit

        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[mix_chosen]
            mat_idx = mix_chosen  # keep in sync with the resolved sub-material (hair needs the real index to re-fetch at connect time)
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            # VCM Stage 2b: finish the per-bounce MIS correction (the
            # dist² portion was already applied above, shared across
            # branches) -- cos_fix is the incoming ray's cosine against
            # this vertex's own normal, matching SmallVCM's cosThetaFix.
            var cos_fix = abs(dot(-ray_dir, gn))
            if cos_fix > Float32(1e-6):
                dvcm_carry /= cos_fix
                dvc_carry /= cos_fix
                dvm_carry /= cos_fix
            # Real image-texture reflectance -- see the matching comment in
            # _bdpt_trace_camera_and_connect's diffuse branch. Also feeds
            # the continuation flux multiply below, not just the stored
            # vertex, since both must agree on this bounce's actual albedo.
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn)
            v.beta = flux
            v.alb = eff_alb
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1); v.med_idx = cur_med_idx
            v.wo = vec3f(-ray_dir)  # VCM Stage 2b: needed for _connect's reverse-pdf eval
            v.wavelengths = wavelengths
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            n_verts += 1
            _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
            # Scatter
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            rd = vec3f(_cosine_hemisphere_sample(gn, u1, u2))
            ro = hit + rd*Float32(0.0002)
            flux *= eff_alb
            # VCM Stage 2b: recursive continuation for the NEXT bounce
            # (cosThetaOut/bsdfDirPdfW simplifies to PI exactly for
            # cosine-weighted diffuse sampling; bsdfRevPdfW reuses cos_fix,
            # the same incoming cosine just used above -- see the memory
            # file for the full derivation).
            var cos_theta_out = abs(dot(rd.to_simd(), gn))
            var bsdf_rev_pdf_w = cos_fix / PI
            var dvc_new = PI * (dvc_carry * bsdf_rev_pdf_w + dvcm_carry + mis_vm_weight_factor)
            var dvm_new = PI * (dvm_carry * bsdf_rev_pdf_w + dvcm_carry * mis_vc_weight_factor + Float32(1))
            var bsdf_dir_pdf_w = cos_theta_out / PI
            dvcm_carry = Float32(1) / bsdf_dir_pdf_w if bsdf_dir_pdf_w > Float32(1e-8) else Float32(0)
            dvc_carry = dvc_new
            dvm_carry = dvm_new

        elif mat.type == MatKind.conductor or mat.type == MatKind.coated_conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                gn_c = sphere_outward_normal(hit, sph_c.center).to_simd()
            else:
                gn_c = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=SIMD[DType.float32, 3](frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=SIMD[DType.float32, 3](frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=mat.albedo, pixel_uv=Float32(0),
            )
            var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
            var bs_c: BxDFSample
            if mat.type == MatKind.conductor:
                bs_c = bxdf_sample_conductor(gc_c, mat, uc1, uc2)
            else:
                var ior_c = mat.emission.r if mat.emission.r > Float32(1) else Float32(1.5)
                var usplit_c = pcg.next_float()
                bs_c = bxdf_sample_coated_conductor(gc_c, mat, ior_c, usplit_c, uc1, uc2)
            if bs_c.is_valid == Int8(0):
                break
            # VCM Stage 2d: rough conductor DOES have a real standalone pdf
            # (bxdf_pdf_conductor_ggx) -- finish the shared dist² correction
            # the same way diffuse does, mirroring
            # _bdpt_trace_camera_and_connect's matching conductor branch.
            var alpha_c = max(mat.roughU, mat.roughV)
            var cos_fix_c = abs(dot(-ray_dir, gn_c))
            if cos_fix_c > Float32(1e-6):
                dvc_carry /= cos_fix_c
                dvm_carry /= cos_fix_c
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c)
                v.beta = flux
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = alpha_c
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                n_verts += 1
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
            flux *= bs_c.f
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)
            var cos_theta_out_c = abs(dot(bs_c.wi, gn_c))
            if bxdf_is_delta(bs_c.flags):
                # VCM: specular bounce -- same reset SmallVCM's own delta-
                # bounce case uses, matches the camera-side conductor branch.
                dvcm_carry = Float32(0)
                dvc_carry *= cos_theta_out_c
                dvm_carry *= cos_theta_out_c
            else:
                var bsdf_dir_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, wo_c, bs_c.wi, alpha_c)
                if bsdf_dir_pdf_w_c > Float32(1e-8):
                    var bsdf_rev_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, bs_c.wi, wo_c, alpha_c)
                    var inv_pdf_c = cos_theta_out_c / bsdf_dir_pdf_w_c
                    var dvc_new_c = inv_pdf_c * (dvc_carry * bsdf_rev_pdf_w_c + dvcm_carry + mis_vm_weight_factor)
                    var dvm_new_c = inv_pdf_c * (dvm_carry * bsdf_rev_pdf_w_c + dvcm_carry * mis_vc_weight_factor + Float32(1))
                    dvcm_carry = Float32(1) / bsdf_dir_pdf_w_c
                    dvc_carry = dvc_new_c
                    dvm_carry = dvm_new_c
                else:
                    dvcm_carry = Float32(0)
                    dvc_carry = Float32(0)
                    dvm_carry = Float32(0)

        elif mat.type == MatKind.hair:
            var curve_idx_h = Int(inter.primId.id1)
            var wo_h = (-rd).to_simd()
            var hc = _hair_precompute(mat, sd.curves, curve_idx_h, inter.v, inter.u, wo_h)
            # VCM Stage 2b: hair has no real standalone pdf either -- same
            # out-of-scope treatment as conductor (see that branch's comment
            # + project_vcm_stage2_mis_derivation memory).
            var cos_fix_h = abs(dot(-ray_dir, hc.geo_normal))
            if cos_fix_h > Float32(1e-6):
                dvc_carry /= cos_fix_h
                dvm_carry /= cos_fix_h
            var v_h = _null_vertex()
            v_h.pos = hit
            v_h.normal = vec3f(hc.geo_normal)
            v_h.beta = flux
            v_h.alb = mat.albedo
            v_h.is_surface = Int32(1); v_h.is_delta = Int32(0); v_h.mat_kind = Int32(2)
            v_h.wo = vec3f(wo_h)
            v_h.mat_idx = Int32(mat_idx)
            v_h.hair_curve_idx = Int32(curve_idx_h)
            v_h.hair_h = inter.u
            v_h.hair_v = inter.v
            v_h.pdf_fwd = Float32(1)
            v_h.med_idx = cur_med_idx
            v_h.wavelengths = wavelengths
            n_verts += 1
            _bdpt_store_lvc_vertex(v_h, lvc, lp_idx, n_verts - 1)
            var (wi_hs, f_hs, pdf_hs, _) = _hair_sample_dir(hc, pcg)
            flux *= f_hs / pdf_hs
            rd = vec3f(wi_hs)
            var hsign = Float32(1) if dot(wi_hs, hc.geo_normal) >= Float32(0) else Float32(-1)
            ro = hit + vec3f(hc.geo_normal) * curve_offset_eps(hc.radius) * hsign
            var cos_theta_out_h = abs(dot(wi_hs, hc.geo_normal))
            dvcm_carry = Float32(0)
            dvc_carry *= cos_theta_out_h
            dvm_carry *= cos_theta_out_h

        elif mat.type == MatKind.measured:
            # Mirrors the camera-side measured branch above, minus the NEE
            # loops (light subpaths don't do NEE against other lights) --
            # see that branch's docstring for the shared-interface rationale.
            var gn_m: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_m = Int(inter.primId.id1)
                var sph_m = sd.spheres[si_m]
                gn_m = sphere_outward_normal(hit, sph_m.center).to_simd()
            else:
                gn_m = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_m, ray_dir) > Float32(0): gn_m = gn_m * Float32(-1)
            if mat.measured_idx < Int32(0):
                break
            var wo_m = (-rd).to_simd()
            var frm_m = Frame.from_z(Vec3f(gn_m[0], gn_m[1], gn_m[2]))
            var tangent_m = SIMD[DType.float32, 3](frm_m.x.x, frm_m.x.y, frm_m.x.z)
            var bitangent_m = SIMD[DType.float32, 3](frm_m.y.x, frm_m.y.y, frm_m.y.z)
            var mb = sd.measuredBrdfs[Int(mat.measured_idx)]
            var wo_l_m = SIMD[DType.float32, 3](dot(wo_m, tangent_m), dot(wo_m, bitangent_m), dot(wo_m, gn_m))
            var uml1 = pcg.next_float(); var uml2 = pcg.next_float()
            var (wi_l_m, f_m, pdf_m, valid_m) = bxdf_sample_measured(mb, wo_l_m, uml1, uml2, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
            if not valid_m or pdf_m <= Float32(0):
                break
            # VCM Stage 2b: measured BxDF DOES have a real standalone pdf
            # (bxdf_pdf_measured, unlike conductor/hair) -- in MIS scope
            # this pass, same real treatment as diffuse (see that branch's
            # comments + project_vcm_stage2_mis_derivation memory).
            var cos_fix_m = abs(dot(-ray_dir, gn_m))
            if cos_fix_m > Float32(1e-6):
                dvcm_carry /= cos_fix_m
                dvc_carry /= cos_fix_m
                dvm_carry /= cos_fix_m
            var v_m = _null_vertex()
            v_m.pos = hit
            v_m.normal = vec3f(gn_m)
            v_m.beta = flux
            v_m.alb = mat.albedo
            v_m.is_surface = Int32(1); v_m.is_delta = Int32(0); v_m.mat_kind = Int32(3)
            v_m.wo = vec3f(wo_m)
            v_m.mat_idx = Int32(mat_idx)
            v_m.pdf_fwd = Float32(1)
            v_m.med_idx = cur_med_idx
            v_m.wavelengths = wavelengths
            v_m.dVCM = dvcm_carry; v_m.dVC = dvc_carry; v_m.dVM = dvm_carry
            n_verts += 1
            _bdpt_store_lvc_vertex(v_m, lvc, lp_idx, n_verts - 1)
            var wi_m = tangent_m * wi_l_m[0] + bitangent_m * wi_l_m[1] + gn_m * wi_l_m[2]
            var wilen_m = dot(wi_m, wi_m)
            if wilen_m > Float32(0):
                wi_m = wi_m * (Float32(1.0) / sqrt(wilen_m))
            var cos_wi_m = dot(wi_m, gn_m)
            if cos_wi_m <= Float32(0):
                break
            flux *= f_m * (cos_wi_m / pdf_m)
            rd = vec3f(wi_m)
            ro = hit + rd*Float32(0.0002)
            # VCM Stage 2b: recursive continuation, real forward/reverse pdf
            # from bxdf_pdf_measured (reverse = same call with wo/wi swapped
            # -- ASSUMED convention, not independently verified against
            # measured_bxdf_eval.mojo's exact semantics; flag if results look
            # wrong on sportscar/measured-material scenes).
            var pdf_rev_m = bxdf_pdf_measured(mb, wi_l_m, wo_l_m)
            var dvc_new_m = (cos_wi_m / pdf_m) * (dvc_carry * pdf_rev_m + dvcm_carry + mis_vm_weight_factor)
            var dvm_new_m = (cos_wi_m / pdf_m) * (dvm_carry * pdf_rev_m + dvcm_carry * mis_vc_weight_factor + Float32(1))
            dvcm_carry = Float32(1) / pdf_m if pdf_m > Float32(1e-8) else Float32(0)
            dvc_carry = dvc_new_m
            dvm_carry = dvm_new_m

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var (new_dir, new_org, _) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, mat.albedo.r, n_lbounces, pcg)
            n_lbounces += 1
            # Light path (TransportMode::Importance): do NOT apply the
            # radiance_scale non-symmetric-scattering correction — it's only
            # for camera/Radiance-mode paths, see _dielectric_bounce's
            # docstring.
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rd = vec3f(new_dir)
            ro = point3f(new_org)
            # VCM Stage 2b: dielectric is genuinely delta/specular (true
            # reflect-or-refract, not an approximation like conductor's
            # VNDF sampling) -- matches SmallVCM's own specular-bounce
            # handling directly (vertexcm.hxx:977-982), no LVC vertex
            # stored either way.
            var cos_fix_d = abs(dot(-ray_dir, gn))
            if cos_fix_d > Float32(1e-6):
                dvc_carry /= cos_fix_d
                dvm_carry /= cos_fix_d
            var cos_theta_out_d = abs(dot(new_dir, gn))
            dvcm_carry = Float32(0)
            dvc_carry *= cos_theta_out_d
            dvm_carry *= cos_theta_out_d

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)
            # VCM Stage 2b: pure medium-boundary pass-through, no direction
            # change/BSDF event -- carry state passes through as already
            # dist²-corrected above (the ray genuinely traveled t_hit), no
            # cosFix/reset applied since there's no real "bounce" here.

        else:
            break

    lvc_path_len[lp_idx] = Int32(n_verts)
    return

# ── BSDF/phase evaluation at a vertex ────────────────────────────────────────

@always_inline
def _eval_conductor_ggx(
    n:     SIMD[DType.float32, 3],
    wo:    SIMD[DType.float32, 3],   # toward this vertex's own predecessor
    wi:    SIMD[DType.float32, 3],   # toward the other connected vertex
    alpha: Float32,
    f0: RGB,
) -> SIMD[DType.float32, 3]:
    """Isotropic GGX (Trowbridge-Reitz) conductor f(wo,wi) × |cos(wi,n)|, for an
    arbitrary (not self-sampled) direction pair — the BDPT connection case that
    bxdf_sample_conductor (a self-sampled-direction-only throughput multiplier)
    can't serve. Schlick Fresnel at the half-vector, height-correlated Smith G2."""
    var cos_o = dot(wo, n)
    var cos_i = dot(wi, n)
    if cos_o <= Float32(0) or cos_i <= Float32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var wh = wo + wi
    var whl = dot(wh, wh)
    if whl <= Float32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    wh = wh * (Float32(1) / sqrt(whl))
    var cos_h = dot(wh, n)
    var cos_wo_h = dot(wo, wh)
    if cos_wo_h < Float32(0): cos_wo_h = -cos_wo_h
    var d = ggx_D(cos_h, alpha)
    var g = ggx_G2(cos_o, cos_i, alpha)
    var one_m = Float32(1) - cos_wo_h
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var fr = f0 + (RGB(Float32(1)) - f0) * schlick
    var k = d * g / (Float32(4) * cos_o * cos_i) * cos_i
    return SIMD[DType.float32, 3](k * fr.r, k * fr.g, k * fr.b)

@always_inline
def _eval_vertex(
    v:   BDPTVertex,
    dir_to_other:  SIMD[DType.float32, 3],   # direction from v toward the other connected vertex
    sd:  SceneDescriptor2_C,
) -> SIMD[DType.float32, 3]:
    """Evaluate BSDF (or phase) × cos at vertex v toward dir_to_other. Takes
    `sd` only for mat_kind=2 (hair), to re-fetch the material + curve data
    _hair_precompute needs (v itself only carries small indices, not the
    full precomputed HairLobeConstants — see BDPTVertex's docstring)."""
    if v.is_delta != Int32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    if v.is_surface == Int32(0):
        # Volume scatter: isotropic phase function 1/(4π), no cosine term
        return SIMD[DType.float32, 3](v.alb.r*INV_FOUR_PI, v.alb.g*INV_FOUR_PI, v.alb.b*INV_FOUR_PI)
    var vn = v.normal.to_simd()
    if v.mat_kind == Int32(1):
        var vwo = v.wo.to_simd()
        return _eval_conductor_ggx(vn, vwo, dir_to_other, v.pdf_bwd, v.alb)
    if v.mat_kind == Int32(2):
        var mat = sd.materials[Int(v.mat_idx)]
        var hc = _hair_precompute(mat, sd.curves, Int(v.hair_curve_idx), v.hair_v, v.hair_h, v.wo.to_simd())
        var (cos_ti, f_val, _) = _hair_eval_lobes(
            dir_to_other, hc.tangent, hc.b_perp, hc.n_perp, hc.phi_o,
            hc.dphi0, hc.dphi1, hc.dphi2,
            hc.cos_tp0_o, hc.sin_tp0_o, hc.cos_tp1_o, hc.sin_tp1_o, hc.cos_tp2_o, hc.sin_tp2_o,
            hc.cos_theta_o, hc.sin_theta_o, hc.inv_vm0, hc.inv_vm1, hc.inv_vm2, hc.mp_c0, hc.mp_c1, hc.mp_c2, hc.s,
            hc.A0, hc.A1, hc.A2, hc.A3, hc.lum0, hc.lum1, hc.lum2, hc.lum3, hc.total_lum,
        )
        return SIMD[DType.float32, 3](f_val.r*cos_ti, f_val.g*cos_ti, f_val.b*cos_ti)
    if v.mat_kind == Int32(3):
        # Measured BxDF is inherently spectral (its tabulated `spectra`
        # tensor is indexed by wavelength) -- bxdf_eval_measured always
        # needs v.wavelengths + the spectral table regardless of whether
        # the caller wanted the plain-RGB or spectral connection path, so
        # this one branch serves both (see _connect's dispatch condition).
        var vmat = sd.materials[Int(v.mat_idx)]
        var mb = sd.measuredBrdfs[Int(vmat.measured_idx)]
        var vwo_m = v.wo.to_simd()
        var frm_ev = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tangent_ev = SIMD[DType.float32, 3](frm_ev.x.x, frm_ev.x.y, frm_ev.x.z)
        var bitangent_ev = SIMD[DType.float32, 3](frm_ev.y.x, frm_ev.y.y, frm_ev.y.z)
        var wo_l_ev = SIMD[DType.float32, 3](dot(vwo_m, tangent_ev), dot(vwo_m, bitangent_ev), dot(vwo_m, vn))
        var wi_l_ev = SIMD[DType.float32, 3](dot(dir_to_other, tangent_ev), dot(dir_to_other, bitangent_ev), dot(dir_to_other, vn))
        var (fr_spec_ev, _) = bxdf_eval_measured(mb, wo_l_ev, wi_l_ev, v.wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
        var (r_ev, g_ev, b_ev) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, fr_spec_ev, v.wavelengths)
        var cos_o_ev = dot(dir_to_other, vn)
        if cos_o_ev < Float32(0): cos_o_ev = -cos_o_ev
        return SIMD[DType.float32, 3](r_ev*cos_o_ev, g_ev*cos_o_ev, b_ev*cos_o_ev)
    # Surface: Lambertian f = alb/π × |cos(wo,n)|
    var cos_o = dot(dir_to_other, vn)
    if cos_o < Float32(0): cos_o = -cos_o
    return SIMD[DType.float32, 3](v.alb.r*INV_PI*cos_o, v.alb.g*INV_PI*cos_o, v.alb.b*INV_PI*cos_o)

# ── Spectral siblings of the two functions above (staged spectral rendering
# rollout, Stage 3 -- see project_spectral_rendering memory). Take
# SpectralHandle's fields DECOMPOSED into individual pointer/int params, NOT
# a single by-value SpectralHandle param -- passing that 6-field struct by
# value across a real Mojo function-call boundary is a confirmed,
# reproducible miscompilation (see spectrum.mojo's comment above
# rgb_to_spectral_sample). Hair (mat_kind=2) is NOT covered here (same
# deliberate exclusion as bxdf.mojo's spectral siblings) -- callers must
# check v.mat_kind != 2 before using these; _connect below does exactly
# that by falling back to the plain RGB _eval_vertex/_eval_conductor_ggx for
# any connection touching a hair vertex.
@always_inline
def _eval_conductor_ggx_spectral(
    n:     SIMD[DType.float32, 3],
    wo:    SIMD[DType.float32, 3],
    wi:    SIMD[DType.float32, 3],
    alpha: Float32,
    f0: RGB,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin],
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin],
    wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Spectral counterpart of _eval_conductor_ggx — same GGX/Schlick math,
    f0 converted to a SpectralSample (reflectance convention) before the
    Schlick blend instead of blending plain RGB triples."""
    var cos_o = dot(wo, n)
    var cos_i = dot(wi, n)
    if cos_o <= Float32(0) or cos_i <= Float32(0):
        return SpectralSample(Float32(0))
    var wh = wo + wi
    var whl = dot(wh, wh)
    if whl <= Float32(0):
        return SpectralSample(Float32(0))
    wh = wh * (Float32(1) / sqrt(whl))
    var cos_h = dot(wh, n)
    var cos_wo_h = dot(wo, wh)
    if cos_wo_h < Float32(0): cos_wo_h = -cos_wo_h
    var d = ggx_D(cos_h, alpha)
    var g = ggx_G2(cos_o, cos_i, alpha)
    var one_m = Float32(1) - cos_wo_h
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var f0_spec = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, f0.r, f0.g, f0.b, wavelengths)
    var fr_spec = f0_spec * (Float32(1) - schlick) + SpectralSample(schlick)
    var k = d * g / (Float32(4) * cos_o * cos_i) * cos_i
    return fr_spec * k

@always_inline
def _eval_vertex_spectral(
    v:   BDPTVertex,
    dir_to_other:  SIMD[DType.float32, 3],
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin],
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin],
    wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Spectral counterpart of _eval_vertex for mat_kind in {diffuse(0),
    conductor(1)} and volume scatter — see this function's docstring at the
    top of this section. Callers must not reach mat_kind=2 (hair) here."""
    if v.is_delta != Int32(0):
        return SpectralSample(Float32(0))
    if v.is_surface == Int32(0):
        var alb_spec = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, v.alb.r, v.alb.g, v.alb.b, wavelengths)
        return alb_spec * INV_FOUR_PI
    var vn = v.normal.to_simd()
    if v.mat_kind == Int32(1):
        var vwo = v.wo.to_simd()
        return _eval_conductor_ggx_spectral(vn, vwo, dir_to_other, v.pdf_bwd, v.alb, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
    # Surface: Lambertian f = alb/π × |cos(wo,n)|
    var cos_o = dot(dir_to_other, vn)
    if cos_o < Float32(0): cos_o = -cos_o
    var alb_spec = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, v.alb.r, v.alb.g, v.alb.b, wavelengths)
    return alb_spec * (INV_PI * cos_o)

# ── Geometry term ─────────────────────────────────────────────────────────────

@always_inline
def _geom_term(
    a: BDPTVertex, b: BDPTVertex,
) -> Float32:
    """Geometry factor G(a,b) = |cos_a| × |cos_b| / dist²."""
    var d3 = b.pos - a.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8): return Float32(0)
    var d = sqrt(dist2)
    var dir = d3.to_simd() / d
    var cos_a: Float32
    if a.is_surface == Int32(1):
        cos_a = dot(dir, a.normal.to_simd())
        if cos_a < Float32(0): cos_a = -cos_a
    else:
        cos_a = Float32(1)   # volume: no cosine
    var cos_b: Float32
    if b.is_surface == Int32(1):
        cos_b = dot(dir, b.normal.to_simd())
        if cos_b < Float32(0): cos_b = -cos_b
    else:
        cos_b = Float32(1)
    return cos_a * cos_b / dist2

@always_inline
def _bdpt_vertex_pdfs(
    v: BDPTVertex, dir_to_other: SIMD[DType.float32, 3], sd: SceneDescriptor2_C,
) -> Tuple[Float32, Float32]:
    """Real forward/reverse solid-angle BSDF pdfs for a VCM-MIS-scoped
    vertex (diffuse mat_kind=0, rough conductor/coated_conductor mat_kind=1,
    measured mat_kind=3 -- light-source vertices use their own cosine-
    weighted-emission formula at the call site instead, see _connect's
    docstring; hair stays out of scope, no VNDF-equivalent pdf exists to
    reuse). Returns (dir_pdf_w, rev_pdf_w):

    - dir_pdf_w: the pdf of v's own BSDF sampling `dir_to_other` as the
      outgoing direction, continuing FROM v.
    - rev_pdf_w: the pdf of sampling v's own `wo` (back toward its
      predecessor) as if the path had instead been traced starting FROM
      `dir_to_other` -- i.e. the same pdf function with wo/wi swapped,
      matching SmallVCM's Pdf(..., adjoint=true) reverse-evaluation
      convention (vertexcm.hxx's SampleScattering).

    For conductor, both directions route through bxdf_pdf_conductor_ggx
    (Heitz 2018 VNDF sampling pdf, already used standalone by shading.mojo's
    NEE-MIS path) -- v.wo/v.normal/v.pdf_bwd (GGX alpha) are exactly the
    fields _eval_vertex's own mat_kind=1 branch already reads, so no new
    per-vertex state is needed. For measured, both directions route through
    bxdf_pdf_measured (Dupuy & Jakob tabulated pdf, bxdfs.cpp:1087-1120) --
    needs a reconstructed local tangent frame (Frame.from_z(v.normal), same
    pattern _eval_vertex's own mat_kind=3 branch already uses) to convert
    world-space wo/dir_to_other into the local-frame directions
    bxdf_pdf_measured expects; `sd` is only needed for this measured lookup
    (sd.materials[v.mat_idx].measured_idx -> sd.measuredBrdfs[...]).
    Measured's PER-BOUNCE dVCM/dVC/dVM carry-through already used this same
    real pdf since Stage 2b (see _bdpt_trace_camera_and_connect's/
    _bdpt_trace_light_path's measured branches) -- this was the one
    remaining gap, connect/merge-TIME weighting, not the recursive update
    itself. Diffuse's cos/π formula is unchanged from Stage 2b. Only ever
    called on vertices already known non-delta (both _connect and
    _bdpt_merge_from_cache reject cv.is_delta/lv.is_delta before reaching
    any weight computation), so bxdf_pdf_conductor_ggx's/bxdf_pdf_measured's
    own internal guards are the only "this direction is impossible under
    the sampling scheme" cases they need to handle."""
    if v.mat_kind == Int32(1):
        var n = v.normal.to_simd()
        var wo = v.wo.to_simd()
        var alpha = v.pdf_bwd
        return (
            bxdf_pdf_conductor_ggx(n, wo, dir_to_other, alpha),
            bxdf_pdf_conductor_ggx(n, dir_to_other, wo, alpha),
        )
    if v.mat_kind == Int32(3):
        var n_m = v.normal.to_simd()
        var frm_m = Frame.from_z(Vec3f(n_m[0], n_m[1], n_m[2]))
        var tangent_m = SIMD[DType.float32, 3](frm_m.x.x, frm_m.x.y, frm_m.x.z)
        var bitangent_m = SIMD[DType.float32, 3](frm_m.y.x, frm_m.y.y, frm_m.y.z)
        var wo_m = v.wo.to_simd()
        var wo_l = SIMD[DType.float32, 3](dot(wo_m, tangent_m), dot(wo_m, bitangent_m), dot(wo_m, n_m))
        var dir_l = SIMD[DType.float32, 3](dot(dir_to_other, tangent_m), dot(dir_to_other, bitangent_m), dot(dir_to_other, n_m))
        var mat_m = sd.materials[Int(v.mat_idx)]
        var mb_m = sd.measuredBrdfs[Int(mat_m.measured_idx)]
        return (
            bxdf_pdf_measured(mb_m, wo_l, dir_l),
            bxdf_pdf_measured(mb_m, dir_l, wo_l),
        )
    var cos_dir = abs(dot(dir_to_other, v.normal.to_simd()))
    var cos_wo = abs(dot(v.wo.to_simd(), v.normal.to_simd()))
    return (cos_dir * INV_PI, cos_wo * INV_PI)

@always_inline
def _bdpt_vertex_mis_scoped(v: BDPTVertex) -> Bool:
    """True for vertex kinds _bdpt_vertex_pdfs has a real pdf for: diffuse
    (mat_kind=0, real surface -- excludes volume vertices, which default
    to mat_kind=0 too, see _connect's matching comment), rough
    conductor/coated_conductor (mat_kind=1, always non-delta by
    construction -- delta conductor bounces are never stored as
    connectible vertices at all, see _bdpt_trace_camera_and_connect's
    conductor branch), and measured (mat_kind=3)."""
    if v.is_surface != Int32(1):
        return False
    return v.mat_kind == Int32(0) or v.mat_kind == Int32(1) or v.mat_kind == Int32(3)

# ── Connect one camera vertex to one light vertex ─────────────────────────────

def _connect(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    sd: SceneDescriptor2_C,
    has_med: Bool,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    mis_vm_weight_factor: Float32,
) -> SIMD[DType.float32, 3]:
    """Evaluate the contribution of connecting cv to lv via a shadow ray.
    Each connection is already a complete, self-normalized estimator of its
    own depth-strategy's contribution (see the module's LVC-BPT docstring);
    the caller sums over every vertex of the paired light path with no
    further scaling.

    VCM Stage 2b/2d/153 (2026-07-10/11): real per-vertex MIS weighting
    (Georgiev et al. 2012 / SmallVCM, see project_vcm_stage2_mis_derivation
    memory) is applied when both endpoints have a genuine standalone pdf --
    diffuse (mat_kind=0), light-source vertices (is_light=1, whose
    cosine-weighted emission profile is mathematically the same shape as
    diffuse), rough conductor/coated_conductor (mat_kind=1, GGX-VNDF pdf),
    and measured (mat_kind=3, tabulated-BRDF pdf via a reconstructed local
    frame). Hair (no VNDF-equivalent pdf for Marschner) and volume
    (isotropic phase, no surface normal) fall through to `weight=1`,
    today's plain unweighted behavior -- a deliberately scoped gap, not a
    silent omission."""
    if cv.is_delta != Int32(0) or lv.is_delta != Int32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    var d3 = lv.pos - cv.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var dist = sqrt(dist2)

    # Determine medium for the shadow segment.
    # Use camera vertex's medium (both should agree in a well-defined scene).
    var seg_med = cv.med_idx
    var Tr = _visible_transmittance(cv.pos, lv.pos, seg_med, sd, scratch)
    if Tr[0] < Float32(1e-7) and Tr[1] < Float32(1e-7) and Tr[2] < Float32(1e-7):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    var dir = d3.to_simd() / dist
    var neg_dir = -dir

    # Spectral connection eval (staged spectral rendering rollout, Stage 3 —
    # see project_spectral_rendering memory): if the table is loaded and
    # neither endpoint is hair (mat_kind=2, deliberately excluded — same as
    # bxdf.mojo's spectral siblings; a hair-touching connection falls back to
    # the plain RGB path below), evaluate BOTH vertices' BSDF/emission as a
    # SpectralSample product at the LIGHT vertex's own stored wavelengths
    # (a valid, unbiased hero-wavelength MC choice, applied consistently per
    # connection — the camera subpath's own wavelength sample is discarded
    # for this one connection's purposes, resolving the LVC's cross-
    # wavelength mismatch since cache vertices come from many different
    # light subpaths, each with its own independent wavelength draw), then
    # convert the PRODUCT (not each factor separately) back to RGB — the
    # whole point of spectral rendering's product-of-spectra accuracy.
    # mat_kind=3 (measured) is ALSO routed to the plain-RGB fallback below,
    # same as hair — _eval_vertex's own mat_kind=3 branch already does the
    # full spectral eval internally (bxdf_eval_measured needs v.wavelengths
    # regardless), so there's no separate _eval_vertex_spectral variant to
    # maintain for it.
    var is_hair_conn = cv.mat_kind == Int32(2) or cv.mat_kind == Int32(3) or (lv.is_light == Int32(0) and (lv.mat_kind == Int32(2) or lv.mat_kind == Int32(3)))
    var f_combined: SIMD[DType.float32, 3]
    if is_hair_conn or sd.spectral.res <= 0:
        # BSDF at camera vertex (toward light)
        var f_cam = _eval_vertex(cv, dir, sd)
        # BSDF at light vertex (toward camera)
        var f_lgt: SIMD[DType.float32, 3]
        if lv.is_light == Int32(1):
            # Light emission: f_lgt = Le (emission radiance, no cosine here).
            # _geom_term already computes cos_l at the light surface, so don't multiply again.
            var ln = lv.normal.to_simd()
            var cos_l = dot(neg_dir, ln)  # emission normal vs direction toward cv
            if cos_l <= Float32(0):
                return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
            f_lgt = SIMD[DType.float32, 3](lv.alb.r, lv.alb.g, lv.alb.b)
        else:
            f_lgt = _eval_vertex(lv, neg_dir, sd)
        f_combined = f_cam * f_lgt
    else:
        var wl = lv.wavelengths
        var f_cam_spec = _eval_vertex_spectral(cv, dir, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
        var f_lgt_spec: SpectralSample
        if lv.is_light == Int32(1):
            var ln = lv.normal.to_simd()
            var cos_l = dot(neg_dir, ln)
            if cos_l <= Float32(0):
                return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
            f_lgt_spec = rgb_illuminant_to_spectral_sample(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, lv.alb.r, lv.alb.g, lv.alb.b, wl)
        else:
            f_lgt_spec = _eval_vertex_spectral(lv, neg_dir, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
        var product_spec = f_cam_spec * f_lgt_spec
        var (pr, pg, pb) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, product_spec, wl)
        f_combined = SIMD[DType.float32, 3](pr, pg, pb)

    # Geometry term G = |cos_cv| × |cos_lv| / dist²
    var G = _geom_term(cv, lv)

    var beta = cv.beta * lv.beta
    var contrib = f_combined * SIMD[DType.float32, 3](G, G, G) * Tr
    contrib[0] *= beta.r
    contrib[1] *= beta.g
    contrib[2] *= beta.b

    # VCM Stage 2b/2d: real MIS weight for diffuse/conductor/light-source
    # connections (see this function's docstring + _bdpt_vertex_pdfs'/
    # project_vcm_stage2_mis_derivation memory for the full derivation and
    # its "not independently verified" caveats). cos_cv/cos_lv reuse the
    # same geometry _geom_term computed internally, recomputed here since
    # that helper doesn't expose them.
    if _bdpt_vertex_mis_scoped(cv) and (lv.is_light == Int32(1) or _bdpt_vertex_mis_scoped(lv)):
        var cos_cv = abs(dot(dir, cv.normal.to_simd()))
        var cos_lv = abs(dot(neg_dir, lv.normal.to_simd()))
        var (camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w) = _bdpt_vertex_pdfs(cv, dir, sd)
        # Light-source vertex: forward and reverse pdf are the SAME
        # cosine-weighted-emission formula (no real "wo" to distinguish a
        # direction from, unlike a genuine BSDF bounce) -- REASONED, not
        # independently verified against a reference light-source-specific
        # connect path.
        var light_bsdf_dir_pdf_w: Float32
        var light_bsdf_rev_pdf_w: Float32
        if lv.is_light == Int32(1):
            light_bsdf_dir_pdf_w = cos_lv / PI
            light_bsdf_rev_pdf_w = cos_lv / PI
        else:
            var (ldp, lrp) = _bdpt_vertex_pdfs(lv, neg_dir, sd)
            light_bsdf_dir_pdf_w = ldp
            light_bsdf_rev_pdf_w = lrp
        var camera_bsdf_dir_pdf_a = camera_bsdf_dir_pdf_w * cos_lv / dist2
        var light_bsdf_dir_pdf_a = light_bsdf_dir_pdf_w * cos_cv / dist2
        var w_light = camera_bsdf_dir_pdf_a * (mis_vm_weight_factor + lv.dVCM + lv.dVC * light_bsdf_rev_pdf_w)
        var w_camera = light_bsdf_dir_pdf_a * (mis_vm_weight_factor + cv.dVCM + cv.dVC * camera_bsdf_rev_pdf_w)
        var mis_weight = Float32(1) / (w_light + Float32(1) + w_camera)
        contrib *= mis_weight

    return contrib

# ── Main BDPT render ──────────────────────────────────────────────────────────

def _bdpt_render_core(
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_spp:    Int,
    n_photons_req: Int,
    verbose:  Bool,
) -> Tuple[UnsafePointer[Float32, MutAnyOrigin], UnsafePointer[Float32, MutAnyOrigin]]:
    """Bidirectional Path Tracing main loop with real VCM connect+merge MIS
    (Light Vertex Cache architecture — see the module docstring above),
    factored out of `vcm_render` (its CLI-facing caller, below) so the CPU
    and GPU entry points share the exact same core. Returns (pixels,
    albedo_pixels), each a caller-owned `n_pix*3` Float32 buffer
    (iso-scaled, max_comp-clamped, NOT yet denoised) — same contract
    `_sppm_render_core` follows.

    Each spp sample traces `n_light_paths_merge = max(n_photons_req, n_pix)`
    light subpaths (task #152's fix — decouples the MERGE side's photon
    budget from n_pix, like SPPM's own `--sppm-photons`; see
    project_vcm_stage2_mis_derivation memory). The first `n_pix` of them
    are, as before, deterministically paired one-per-pixel with that
    pixel's own camera subpath for CONNECTION (see
    _bdpt_store_lvc_vertex's docstring) — this pairing is untouched by
    `n_photons_req`, since `_bdpt_connect_to_cache` only ever reads
    `lvc[pix]`, never `n_light_paths_merge`. Any EXTRA light paths beyond
    n_pix exist purely to densify the merge side's spatial hash grid — the
    merge normalization (`merge_norm`/`eta_vcm` below) uses the TRUE total
    `n_light_paths_merge`, so the per-vertex MIS weights stay correct (the
    balance heuristic only needs each technique's real sampling density,
    which is well-defined for any n_light_paths_merge >= n_pix, not
    specifically n_light_paths_merge == n_pix). Then traces every pixel's
    camera subpath and, at each non-delta vertex, both connects to every
    vertex of its paired light path AND merges against all light paths'
    vertices (now including the extra merge-only ones) within the current
    sample's progressive merge radius —
    see this file's opening VCM comment for the combined estimator and
    `_bdpt_connect_to_cache`/`_bdpt_merge_from_cache`'s own docstrings for
    each technique's per-candidate MIS weight."""
    var fw = Int(psc[0].film_w)
    var fh = Int(psc[0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[0].film_iso / Float32(100)
    var max_comp  = psc[0].film_max_comp
    # VCM Stage 2b: world-space size of one pixel at unit distance along the
    # camera forward axis -- same quantity the plain path tracer's mip LOD
    # uses (pipeline.mojo), reused here for the camera-origin cameraPdfW
    # derivation (see project_vcm_stage2_mis_derivation memory).
    var px_scale = Float32(2.0) * tan(psc[0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(fh)

    var n_light_paths_merge = max(n_photons_req, n_pix)
    print("VCM: " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
          + String(n_light_paths_merge) + " light paths/pass")

    var has_med = Int(sd.mediumCount) > 0

    # Determine starting medium for light subpaths (same logic as SPPM)
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # Output buffer: one RGB per pixel, plus a parallel first-hit-albedo AOV
    # accumulator for the post-render denoiser (see write_image call below).
    var buf = alloc[RGB](n_pix)
    var albedo_buf = alloc[RGB](n_pix)
    for i in range(n_pix):
        buf[i] = RGB(Float32(0))
        albedo_buf[i] = RGB(Float32(0))

    var r2c = psc[0].raster_to_camera
    var c2w = psc[0].camera_to_world
    var base_seed = psc[0].rng_seed

    # The first n_pix light paths are DETERMINISTICALLY paired with that
    # pixel's eye subpath, standard Veach BDPT pairing -- see
    # _bdpt_store_lvc_vertex's docstring and project_vcm_stage2_mis_derivation
    # memory. Any paths beyond n_pix (n_light_paths_merge > n_pix, task
    # #152's photon-budget decoupling) exist only to densify the merge
    # grid -- they're never read by any pixel's connect step, only by
    # _bdpt_merge_from_cache's spatial-grid walk. Each light path owns its
    # own dedicated _BDPT_MAX_VERTS-sized slice of `lvc` (no shared-pool
    # contention, no atomics); `lvc_path_len[lp_idx]` records how many of
    # those slots it actually filled.
    var lvc_cap = n_light_paths_merge * _BDPT_MAX_VERTS
    var lvc = alloc[BDPTVertex](max(lvc_cap, 1))
    var lvc_path_len = alloc[Int32](max(n_light_paths_merge, 1))
    # One scratch Intersection_C per concurrent worker (light path / pixel)
    # instead of one shared slot — CPU threads now race on this exactly like
    # GPU threads already do (see _bdpt_emit_light_paths_gpu/
    # _bdpt_camera_connect_gpu's own per-thread inter_light_ptr+k/
    # inter_cam_ptr+pix), so it can no longer be a single reused buffer.
    var scratch_light = alloc[Intersection_C](max(n_light_paths_merge, 1))
    var scratch_cam = alloc[Intersection_C](max(n_pix, 1))

    # VCM vertex merging: grid buffers allocated once, rebuilt fresh every
    # spp sample (mirrors the LVC itself). Stage 2c: the radius itself is
    # now progressive (Hachisuka & Jensen 2008's global, per-iteration
    # scheme -- see this file's opening VCM comment for why that's the
    # right choice here, not SPPM's per-pixel adaptive radius), recomputed
    # each `si` below from `merge_radius_1`, the same initial 3%-of-scene-
    # diameter value Stage 1 used as its (then-fixed) radius.
    var (_scene_center, scene_radius) = _scene_bounding_sphere(sd)
    var merge_radius_1 = scene_radius * Float32(0.03)
    comptime _VCM_RADIUS_ALPHA = Float32(2.0) / Float32(3.0)  # Georgiev 2012's typical choice
    var merge_heads = alloc[Int32](_HSIZE)
    var merge_next = alloc[Int32](max(lvc_cap, 1))

    for si in range(n_spp):
        # Stage 2c progressive radius: r_i = r_1 / (i+1)^(0.5*(1-alpha))
        # (Hachisuka & Jensen 2008 via Georgiev et al. 2012 Eq. 11), a
        # single GLOBAL radius shared by every pixel this sample, shrinking
        # monotonically across samples -- distinct from sppm.mojo's
        # per-pixel Knaus-Zwicker scheme (see this file's opening comment).
        var radius_i = merge_radius_1 / pow(Float32(si + 1), Float32(0.5) * (Float32(1) - _VCM_RADIUS_ALPHA))
        var merge_r2 = radius_i * radius_i
        var merge_inv_cell = Float32(1.0) / max(radius_i, Float32(1e-6))
        var merge_norm = Float32(1.0) / (Float32(n_light_paths_merge) * PI * max(merge_r2, Float32(1e-12)))

        # VCM Stage 2b/2c: global per-iteration MIS weight-combination
        # constants (Georgiev et al. 2012 / SmallVCM, see
        # project_vcm_stage2_mis_derivation memory), recomputed every
        # sample since they depend on the now-progressive radius. Merge
        # runs unconditionally alongside connect (see this file's opening
        # VCM comment), so mis_vm_weight_factor uses its real,
        # non-discounted eta_vcm-derived value. Task #152: eta_vcm must use
        # the TRUE total merge-candidate pool size (n_light_paths_merge),
        # not n_pix -- the balance heuristic weight is only correct if it
        # reflects each technique's actual sampling density, and merge's
        # density scales with however many light paths actually feed its
        # grid, independent of how many camera pixels exist.
        var eta_vcm = PI * max(merge_r2, Float32(1e-12)) * Float32(n_light_paths_merge)
        var mis_vm_weight_factor = eta_vcm
        var mis_vc_weight_factor = Float32(1.0) / eta_vcm

        # ── Phase 1: trace every light subpath (n_light_paths_merge total,
        # the first n_pix of them pixel-paired, see above) ───────────────────
        # No atomics needed: light path lp_idx writes only its own dedicated
        # slice of lvc (see _bdpt_store_lvc_vertex's docstring).
        @parameter
        def emit_light_path(lp_idx: Int):
            var lpcg = PCG32(base_seed ^ UInt64(lp_idx * 6364136223846793005 + 1442695040888963407),
                              UInt64(si * 2654435761 + 1))
            _bdpt_trace_light_path[False](sd, lpcg, has_med, default_emit_med,
                                         scratch_light + lp_idx, lvc, lp_idx, lvc_path_len,
                                         mis_vc_weight_factor, mis_vm_weight_factor)

        parallelize[emit_light_path](n_light_paths_merge)

        _bdpt_build_merge_grid(lvc, lvc_path_len, n_light_paths_merge, merge_next, merge_heads, merge_inv_cell)

        # ── Phase 2: trace each pixel's camera path and connect ──────────────
        # Each worker only ever writes its own buf[pix] slot and only reads
        # (never mutates) the now-fully-built lvc cache — no atomics needed.
        # pix doubles as this pixel's PAIRED light path index (VCM Stage 2b,
        # the first n_pix of n_light_paths_merge total light paths, one
        # dedicated light path per pixel -- unaffected by task #152's extra
        # merge-only paths beyond n_pix).
        @parameter
        def camera_connect(pix: Int):
            var px = pix % fw; var py = pix // fw
            var cpcg = PCG32(base_seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                              UInt64(si * 2654435761 + 1))
            var (contrib, alb) = _bdpt_trace_camera_and_connect[False](
                r2c, c2w, px, py, sd, cpcg, has_med, scratch_cam + pix, lvc, pix, Int(lvc_path_len[pix]),
                merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
                px_scale, mis_vc_weight_factor, mis_vm_weight_factor, Float32(n_light_paths_merge))
            buf[pix] += contrib
            albedo_buf[pix] += alb

        parallelize[camera_connect](n_pix)

        if verbose:
            print("VCM: sample " + String(si + 1) + "/" + String(n_spp))

    scratch_light.free(); scratch_cam.free(); lvc.free(); lvc_path_len.free()
    merge_heads.free(); merge_next.free()

    # Clamp into caller-owned output buffers (no denoise/write here -- see
    # vcm_render/vcm_render_gpu, this function's two callers, for the tail).
    var inv_spp = iso_scale / Float32(n_spp)
    var pixels = alloc[Float32](n_pix * 3)
    for i in range(n_pix):
        var c = buf[i] * inv_spp
        if max_comp > Float32(0):
            c.r = c.r if c.r < max_comp else max_comp
            c.g = c.g if c.g < max_comp else max_comp
            c.b = c.b if c.b < max_comp else max_comp
        pixels[i*3]   = c.r
        pixels[i*3+1] = c.g
        pixels[i*3+2] = c.b
    buf.free()

    var albedo_pixels = alloc[Float32](n_pix * 3)
    var inv_spp_alb = Float32(1) / Float32(n_spp)
    for i in range(n_pix):
        var a = albedo_buf[i] * inv_spp_alb
        albedo_pixels[i*3]   = a.r
        albedo_pixels[i*3+1] = a.g
        albedo_pixels[i*3+2] = a.b
    albedo_buf.free()

    return Tuple[UnsafePointer[Float32, MutAnyOrigin], UnsafePointer[Float32, MutAnyOrigin]](pixels, albedo_pixels)

def vcm_render(
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_spp:    Int,
    n_photons: Int,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """CLI-facing VCM entry point: run the real connect+merge per-vertex-MIS
    estimator (`_bdpt_render_core` -- VCM Stage 2b/2c, see the module's
    opening VCM comment), then denoise (first-hit albedo AOV + a fresh
    unjittered normals/depth pass via the SAME render_aux_buffers the plain
    path tracer uses -- integrator-agnostic, no dependency on this
    estimator's own path state) and write. `n_photons` (task #152,
    already resolved by pipeline.mojo's `_resolve_vcm_photons` --
    `--vcm-photons` if given, else n_pix) is the merge side's light-path
    budget per pass, decoupled from n_pix; see _bdpt_render_core's
    docstring."""
    var n_pix = Int(psc[0].film_w) * Int(psc[0].film_h)
    var (pixels, albedo_pixels) = _bdpt_render_core(psc, sd, n_spp, n_photons, verbose)

    var normals = alloc[Float32](n_pix * 3)
    var depth = alloc[Float32](n_pix)
    var sd_local = sd
    render_aux_buffers(psc[0].raster_to_camera, psc[0].camera_to_world, Int32(0), Int32(0),
                        psc[0].film_w, psc[0].film_h, UnsafePointer(to=sd_local), normals, depth)

    var denoised = alloc[Float32](n_pix * 3)
    if no_denoise:
        for i in range(n_pix * 3): denoised[i] = pixels[i]
    else:
        denoise(pixels, albedo_pixels, normals, depth, psc[0].film_w, psc[0].film_h,
                denoised, Int32(5), Float32(3.0), Float32(0.2), Float32(0.3), Float32(0.05))

    _ = write_image(denoised, psc[0].film_w, psc[0].film_h, psc[0].film_filename, Int32(32), Int32(32))
    pixels.free(); albedo_pixels.free(); normals.free(); depth.free(); denoised.free()
    return Int32(0)

# ── GPU port ───────────────────────────────────────────────────────────────
# Everything below reuses _bdpt_trace_light_path[True]/_bdpt_trace_camera_
# and_connect[True] verbatim — the SAME functions vcm_render (CPU) calls
# with [False] above. This is deliberately unlike the OLD gpu_sppm.mojo's
# GPU port of sppm.mojo (a full line-by-line reimplementation of every
# bounce loop) — sppm.mojo has since been retrofitted to the same
# comptime[use_gpu] pattern and gpu_sppm.mojo deleted (see
# project_unified_renderer_roadmap in memory) — the whole point of doing
# BDPT's port this way first is to prove the zero-duplication comptime[use_gpu] pattern
# (already used for the wavefront path tracer's shade_nee_core) scales to a
# full renderer, as a template for eventually retrofitting SPPM the same way.

# ── Kernels ───────────────────────────────────────────────────────────────
# _mk_sd_full (builds a complete SceneDescriptor2_C from raw GPU device
# pointers) now lives in bvh.mojo, next to SceneDescriptor2_C itself, since
# sppm.mojo's own GPU kernels need the exact same helper and importing it
# from here would create an import cycle (bdpt.mojo already imports shared
# helpers from .sppm).

def _bdpt_emit_light_paths_gpu(
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_path_len: UnsafePointer[Int32, MutAnyOrigin],
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    inter_scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    n_light_paths: Int,
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx: Int,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int64,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    pointLightCount: Int64,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    measuredBrdfs: UnsafePointer[MeasuredBRDF_C, MutAnyOrigin] = UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: UnsafePointer[GpuTexture_C, MutAnyOrigin] = UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
):
    """One thread per light path, each writing only its own dedicated
    per-path slice of `lvc` (VCM Stage 2b, see _bdpt_store_lvc_vertex's
    docstring) -- no atomics/contention. Thin wrapper: build sd, seed this
    thread's own PCG32, call the SAME _bdpt_trace_light_path vcm_render's
    CPU driver calls (with [False] on CPU, [True] here). `has_med` isn't a
    kernel parameter (`Bool` isn't a `DevicePassable` type `enqueue_function`
    accepts) -- derived here from `mediumCount`, which already is."""
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_light_paths:
        return
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
        distantLights, distantLightCount, infiniteLights, infiniteLightCount,
        pointLights, pointLightCount,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        measuredBrdfs, measuredBrdfCount,
        gpuTextures, gpuTextureCount,
    )
    var has_med = mediumCount > Int64(0)
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    var scratch = inter_scratch + k
    _bdpt_trace_light_path[True](sd, pcg, has_med, default_emit_med, scratch, lvc, k, lvc_path_len,
                                 mis_vc_weight_factor, mis_vm_weight_factor)

def _bdpt_camera_connect_gpu(
    accum: UnsafePointer[Float32, MutAnyOrigin],
    albedo_accum: UnsafePointer[Float32, MutAnyOrigin],
    n_pix: Int,
    fw: Int,
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    inter_scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_path_len: UnsafePointer[Int32, MutAnyOrigin],
    merge_next: UnsafePointer[Int32, MutAnyOrigin],
    merge_heads: UnsafePointer[Int32, MutAnyOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    px_scale: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_f: Float32,
    seed: UInt64,
    pass_idx: Int,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int64,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    pointLightCount: Int64,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    measuredBrdfs: UnsafePointer[MeasuredBRDF_C, MutAnyOrigin] = UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: UnsafePointer[GpuTexture_C, MutAnyOrigin] = UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
):
    """One thread per pixel. Thin wrapper: build sd, seed this thread's own
    PCG32 (same seed formula vcm_render's CPU driver uses, keyed by pixel
    index), call the SAME _bdpt_trace_camera_and_connect with [True], then
    accumulate straight into this pixel's own slot of `accum` — race-free
    since every thread owns exactly one pixel, the same reasoning the live
    (non-queued) shadow-ray path in gpu.mojo's shade_*_gpu kernels already
    relies on. `has_med` isn't a kernel parameter (see
    _bdpt_emit_light_paths_gpu's docstring) -- derived from mediumCount.
    merge_* params carry this pass's progressive-radius merge grid (VCM
    Stage 2c, see the module's opening VCM comment) -- the grid is built
    once per pass by vcm_render_gpu before this kernel launches, mirroring
    the LVC's own build-then-consume shape."""
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
        distantLights, distantLightCount, infiniteLights, infiniteLightCount,
        pointLights, pointLightCount,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        measuredBrdfs, measuredBrdfCount,
        gpuTextures, gpuTextureCount,
    )
    var has_med = mediumCount > Int64(0)
    var px = pix % fw
    var py = pix // fw
    var pcg = PCG32(seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                     UInt64(pass_idx * 2654435761 + 1))
    var scratch = inter_scratch + pix
    var (contrib, alb) = _bdpt_trace_camera_and_connect[True](
        r2c, c2w, px, py, sd, pcg, has_med, scratch, lvc, pix, Int(lvc_path_len[pix]),
        merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
        px_scale, mis_vc_weight_factor, mis_vm_weight_factor, n_light_paths_f)
    accum[pix*3]   += contrib.r
    accum[pix*3+1] += contrib.g
    accum[pix*3+2] += contrib.b
    albedo_accum[pix*3]   += alb.r
    albedo_accum[pix*3+1] += alb.g
    albedo_accum[pix*3+2] += alb.b

def bdpt_merge_grid_reset_gpu(heads: UnsafePointer[Int32, MutAnyOrigin], hsize: Int):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= hsize:
        return
    _bdpt_reset_merge_cell(heads, tid)


def bdpt_merge_grid_insert_gpu(
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_path_len: UnsafePointer[Int32, MutAnyOrigin],
    lvc_cap: Int,
    merge_next: UnsafePointer[Int32, MutAnyOrigin],
    heads: UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= lvc_cap:
        return
    _bdpt_insert_merge_vertex[True](k, lvc, lvc_path_len, merge_next, heads, inv_cell)


# ── Host driver ───────────────────────────────────────────────────────────

def vcm_render_gpu(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_spp:    Int,
    n_photons_req: Int,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """GPU-accelerated Light Vertex Cache BDPT — same algorithm as
    vcm_render (CPU), same shared _bdpt_trace_light_path/
    _bdpt_trace_camera_and_connect functions, parallelized: one thread per
    light path for the light pass, one thread per pixel for the camera+
    connect pass. Mirrors sppm.mojo's sppm_render_gpu per-pass
    reset-counter -> emit -> sync+readback+clamp -> consume shape.
    `n_photons_req` (task #152) is the merge side's requested light-path
    budget, decoupled from n_pix -- see _bdpt_render_core's (CPU)
    docstring for the full derivation; this function mirrors that same
    n_light_paths_merge = max(n_photons_req, n_pix) split."""
    var fw = Int(psc[0].film_w)
    var fh = Int(psc[0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[0].film_iso / Float32(100)
    var max_comp  = psc[0].film_max_comp
    var n_light_paths_merge = max(n_photons_req, n_pix)

    print("VCM (GPU): " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
          + String(n_light_paths_merge) + " light paths/pass")

    var has_med = Int(sd.mediumCount) > 0
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    var lvc_cap = n_light_paths_merge * _BDPT_MAX_VERTS
    var base_seed = psc[0].rng_seed

    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256

            var lvc_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](max(lvc_cap, 1) * size_of[BDPTVertex]())
            var path_len_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths_merge, 1) * size_of[Int32]())
            # VCM vertex merging (Stage 1) grid buffers — see vcm_render's
            # matching CPU allocation for the merge_r2/merge_norm derivation.
            var merge_heads_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](_HSIZE * size_of[Int32]())
            var merge_next_buf  = handle[].ctx.enqueue_create_buffer[DType.uint8](max(lvc_cap, 1) * size_of[Int32]())
            var inter_light_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths_merge, 1) * size_of[Intersection_C]())
            var inter_cam_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[Intersection_C]())
            var accum_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[i] = Float32(0)
            var albedo_accum_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with albedo_accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[i] = Float32(0)

            var r2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with r2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[0].raster_to_camera.bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[i] = src[i]
            var c2w_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with c2w_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[0].camera_to_world.bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[i] = src[i]

            var lvc_ptr     = lvc_buf.unsafe_ptr().bitcast[BDPTVertex]()
            var path_len_ptr = path_len_buf.unsafe_ptr().bitcast[Int32]()
            var merge_heads_ptr = merge_heads_buf.unsafe_ptr().bitcast[Int32]()
            var merge_next_ptr  = merge_next_buf.unsafe_ptr().bitcast[Int32]()
            var inter_light_ptr = inter_light_buf.unsafe_ptr().bitcast[Intersection_C]()
            var inter_cam_ptr   = inter_cam_buf.unsafe_ptr().bitcast[Intersection_C]()
            var accum_ptr   = accum_buf.unsafe_ptr().bitcast[Float32]()
            var albedo_accum_ptr = albedo_accum_buf.unsafe_ptr().bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().bitcast[Float32]()

            var bvh2Nodes = handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node]()
            var primIds = handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C]()
            var meshes = handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C]()
            var curves = handle[].curves_buf.unsafe_ptr().bitcast[Curve_C]()
            var blasNodesArr = handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]]()
            var blasPrimIdsArr = handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]]()
            var instances = handle[].instances_buf.unsafe_ptr().bitcast[Instance_C]()
            var materials = handle[].materials_buf.unsafe_ptr().bitcast[Material_C]()
            var mediums = handle[].mediums_buf.unsafe_ptr().bitcast[Medium_C]()
            var mediumInterfaces = handle[].medium_ifaces_buf.unsafe_ptr().bitcast[MediumInterface_C]()
            var spheres = handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C]()
            var areaLights = handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C]()
            var distantLights = handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C]()
            var infiniteLights = handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C]()
            var pointLights = handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C]()
            var n_mediums = Int64(handle[].n_mediums)
            var n_medium_ifaces = Int64(handle[].n_medium_ifaces)
            var n_spheres = Int64(handle[].n_spheres)
            var n_curves = Int64(handle[].n_curves)
            var n_area_lights = Int64(handle[].n_area_lights)
            var n_distant_lights = Int64(handle[].n_distant_lights)
            var n_infinite_lights = Int64(handle[].n_infinite_lights)
            var n_point_lights = Int64(handle[].n_point_lights)
            var n_blas = Int64(handle[].n_blas)
            var n_instances = Int64(handle[].n_instances)
            var spectral_coeffs = handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32]()
            var spectral_cie_x = handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32]()
            var spectral_cie_y = handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32]()
            var spectral_cie_z = handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32]()
            var spectral_d65 = handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32]()
            var spectral_res = handle[].spectral_res
            var measured_brdfs = handle[].measured_brdfs_buf.unsafe_ptr().bitcast[MeasuredBRDF_C]()
            var n_measured_brdfs = Int64(handle[].n_measured_brdfs)
            var gpu_textures = handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C]()
            var n_gpu_textures = Int64(handle[].n_textures)

            var grid_light = ceildiv(max(n_light_paths_merge, 1), block_size)
            var grid_pix = ceildiv(n_pix, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            var (_scene_center, scene_radius) = _scene_bounding_sphere(sd)
            var merge_radius_1 = scene_radius * Float32(0.03)
            comptime _VCM_RADIUS_ALPHA = Float32(2.0) / Float32(3.0)
            var px_scale = Float32(2.0) * tan(psc[0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(fh)
            var n_light_paths_f = Float32(n_light_paths_merge)

            var grid_merge_ins = ceildiv(max(lvc_cap, 1), block_size)

            for si in range(n_spp):
                # Stage 2c progressive radius -- see vcm_render (CPU)'s
                # matching per-sample loop for the full derivation comment.
                var radius_i = merge_radius_1 / pow(Float32(si + 1), Float32(0.5) * (Float32(1) - _VCM_RADIUS_ALPHA))
                var merge_r2 = radius_i * radius_i
                var merge_inv_cell = Float32(1.0) / max(radius_i, Float32(1e-6))
                var merge_norm = Float32(1.0) / (Float32(n_light_paths_merge) * PI * max(merge_r2, Float32(1e-12)))
                var eta_vcm = PI * max(merge_r2, Float32(1e-12)) * Float32(n_light_paths_merge)
                var mis_vm_weight_factor = eta_vcm
                var mis_vc_weight_factor = Float32(1.0) / eta_vcm

                var pass_seed = base_seed ^ UInt64(si * 2654435761 + 1)
                handle[].ctx.enqueue_function[_bdpt_emit_light_paths_gpu](
                    lvc_ptr, path_len_ptr, mis_vc_weight_factor, mis_vm_weight_factor,
                    inter_light_ptr, n_light_paths_merge,
                    default_emit_med, pass_seed, si,
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grid_dim=grid_light, block_dim=block_size)

                # VCM Stage 2b: light paths are deterministically paired with
                # pixels (the first n_pix of n_light_paths_merge total light
                # paths, task #152), so no host readback of a total vertex
                # count is needed anymore -- lvc_cap is already known at
                # compile/host time. Kernels stay ordered on one stream
                # without an explicit synchronize() here.
                handle[].ctx.enqueue_function[bdpt_merge_grid_reset_gpu](
                    merge_heads_ptr, _HSIZE, grid_dim=grid_hsize, block_dim=block_size)
                handle[].ctx.enqueue_function[bdpt_merge_grid_insert_gpu](
                    lvc_ptr, path_len_ptr, lvc_cap, merge_next_ptr, merge_heads_ptr, merge_inv_cell,
                    grid_dim=grid_merge_ins, block_dim=block_size)

                handle[].ctx.enqueue_function[_bdpt_camera_connect_gpu](
                    accum_ptr, albedo_accum_ptr, n_pix, Int(psc[0].film_w), r2c_ptr, c2w_ptr, inter_cam_ptr,
                    lvc_ptr, path_len_ptr,
                    merge_next_ptr, merge_heads_ptr, merge_inv_cell, merge_r2, merge_norm,
                    px_scale, mis_vc_weight_factor, mis_vm_weight_factor, n_light_paths_f,
                    base_seed, si,
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grid_dim=grid_pix, block_dim=block_size)

                if verbose:
                    print("VCM (GPU): sample " + String(si + 1) + "/" + String(n_spp))

            handle[].ctx.synchronize()

            var pixels = alloc[Float32](n_pix * 3)
            with accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().bitcast[Float32]()
                var inv_spp = iso_scale / Float32(n_spp)
                for i in range(n_pix):
                    var r = src[i*3]   * inv_spp
                    var g = src[i*3+1] * inv_spp
                    var b = src[i*3+2] * inv_spp
                    if max_comp > Float32(0):
                        r = r if r < max_comp else max_comp
                        g = g if g < max_comp else max_comp
                        b = b if b < max_comp else max_comp
                    pixels[i*3] = r; pixels[i*3+1] = g; pixels[i*3+2] = b

            # Denoise (never wired up before -- no_denoise was a dead
            # parameter): read back the albedo AOV accumulated above, run
            # a fresh normals/depth pass via the host-side sd (same
            # render_aux_buffers the CPU path/plain tracer use -- host-only,
            # so it runs on the CPU here too, not as a GPU kernel), then the
            # same CPU denoise() the CPU BDPT path uses.
            var albedo_pixels = alloc[Float32](n_pix * 3)
            with albedo_accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().bitcast[Float32]()
                var inv_spp_alb = Float32(1) / Float32(n_spp)
                for i in range(n_pix * 3):
                    albedo_pixels[i] = src[i] * inv_spp_alb

            var normals = alloc[Float32](n_pix * 3)
            var depth = alloc[Float32](n_pix)
            var sd_local = sd
            render_aux_buffers(psc[0].raster_to_camera, psc[0].camera_to_world, Int32(0), Int32(0),
                                psc[0].film_w, psc[0].film_h, UnsafePointer(to=sd_local), normals, depth)

            var denoised = alloc[Float32](n_pix * 3)
            if no_denoise:
                for i in range(n_pix * 3): denoised[i] = pixels[i]
            else:
                denoise(pixels, albedo_pixels, normals, depth, psc[0].film_w, psc[0].film_h,
                        denoised, Int32(5), Float32(3.0), Float32(0.2), Float32(0.3), Float32(0.05))

            _ = write_image(denoised, psc[0].film_w, psc[0].film_h, psc[0].film_filename, Int32(32), Int32(32))
            pixels.free(); albedo_pixels.free(); normals.free(); depth.free(); denoised.free()
        except e:
            print("VCM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("VCM GPU: no accelerator")
        ret = Int32(-1)
    return ret
