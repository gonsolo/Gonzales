# Bidirectional Path Tracing (CPU, multithreaded, + GPU).
# Supports homogeneous participating media and specular chains (glass).
# Strategies: t >= 1, s >= 1 only (no lens sampling for s=0).
# MIS: balance heuristic over all valid connection strategies.

from std.sys import has_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import parallelize
from std.math import sqrt, cos, sin, tan, floor, log, exp, max, min, abs, ceildiv, pow
from std.memory.alloc import unsafe_alloc
from std.atomic import Atomic
from .geometry import (
    RGB, Point3f, Point2f, Vec3f, vec3f, point3f, Ray_C, Intersection_C, Frame,
    TriangleMesh_C, Material_C, MatKind, LobeKind, PhotonKind, AreaLight_C, Medium_C, MediumInterface_C,
    Sphere_C, Curve_C, PrimId_C, Instance_C, DistantLight_C, InfiniteLight_C, PointLight_C,
    MeasuredBRDF_C, GpuTexture_C,
    dot, cross, fr_dielectric, sphere_outward_normal, refract, PI, INV_FOUR_PI, INV_PI,
    PDF_DROP_DIRECT, PDF_VOL_PHASE_HIT,
    cos_theta_t_dielectric, coat_beer_lambert_tr, DEFAULT_COAT_THICKNESS,
    FreeFlight, sample_homogeneous_free_flight, sample_free_flight, medium_is_heterogeneous, medium_sigma_t_spectral, SSS_WALK_ROUNDS,
    Grid_C, NvdbGrid_C,
    spectral_free_flight_weight,
)
from .bssrdf import dipole_max_radius, dipole_rd, dipole_mis_sigma_tr, dipole_sample_radius, bssrdf_probe_offset, bssrdf_exit_pdf_area, bssrdf_exit_ft, fdr_moment
from .vcm_mis import mis_policy_power, vcm_arrival_carries, vcm_scatter_carries, bssrdf_hop_carries, bssrdf_exit_scatter_carries, vcm_env_nee_weight, vcm_env_escape_weight, MisPolicy, nee_mis_weight
from .bvh import (
    BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core, test_spheres, _mk_sd_full,
    _scene_bounding_sphere, _sample_disk_perpendicular, _sample_infinite_light_dir, _eval_infinite_light_and_pdf, _is_real_ptr,
    HairLobeConstants, _hair_precompute, _hair_eval_lobes, _hair_sample_dir, curve_offset_eps,
    LightSample, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, _sample_infinite_light_nee,
    render_aux_buffers,
)
from .sampling import power_heuristic, sample_ggx_vndf, sample_cosine_hemisphere_world, mix_bits_u64, camera_ray_from_film_xy, \
    FilmFilter, film_filter_of, film_filter_offset, filter_eval_2d, filter_integral_2d
from .rng import PCG32
from .transform import matrix_invert
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image, write_image_cropwindow, denoise
from .sppm import _geom_normal, _dielectric_bounce, medium_after_crossing, _cosine_hemisphere_sample, sample_area_light_uniform, _HSIZE, _hash_cell, _sppm_render_core
from .sppm import (
    SPPMPixel, SPPMPhoton, _sppm_reset_grid_cell, _sppm_insert_photon,
    _sppm_gather_one, _sppm_vp_brdf, _sppm_nee_one,
    _sppm_finalize_albedo_one_pixel, _sppm_finalize_one_pixel,
    _VP_SAMPLES, _sppm_has_sphere_lights, _MAX_B,
    _sppm_trace_visible_point, _sppm_store_photon, _sppm_trace_photon,
    _sppm_cam_pos, _sppm_photon_px_scale, gather_disk_contains,
)
from .shading import _tex_lookup, _get_tri_verts, _mnee_walk, _mnee_walk2, \
    apply_surface_maps_at_hit, _camera_approx_footprint, area_light_hit_cos, curve_light_hit
from .bxdf import LobeCtx, LobeEval, lobe_eval, lobe_scoped, _eval_conductor_ggx_spectral, coat_eval_smooth, bxdf_pdf_coated_exit, CoatWalk, coat_walk_begin, coat_walk_enter, coat_walk_at_base, coat_walk_scatter, COAT_WALKING, COAT_REFLECT, COAT_EXIT, COAT_ABSORB, GeomContext, BxDFSample, bxdf_sample_conductor, bxdf_sample_coated_conductor, bxdf_is_delta, bxdf_eval_diffuse, bxdf_pdf_diffuse, ggx_D, ggx_G1, ggx_G2, ggx_vndf_pdf, bxdf_eval_conductor_ggx, bxdf_pdf_conductor_ggx, _nee_weight_simple, _nee_weight_hair, _nee_weight_simple_spectral, _nee_weight_coated_coat_lobe, _nee_weight_coated_diffuse_base, LobeTables
from .measured_bxdf_eval import bxdf_eval_measured, bxdf_sample_measured, _nee_weight_measured, bxdf_pdf_measured
from .gpu import GpuSceneHandle, vulkaninterop_unpack_results_kernel
from .vulkaninterop import VulkanInteropRtSceneHandle, vulkaninterop_rt_trace
from max.gpu.host._nvidia_cuda import CUDA
from .spectrum import (
    SampledWavelengths, SpectralSample, sample_wavelengths_uniform, SpectralHandle,
    pass_wavelengths,
    spec_refl, spec_refl_unbounded, spec_illum,
    rgb_to_spectral_sample, rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb,
    rgb_bands_to_spectral_sample,
)

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
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian,
        wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

# ── Geometry helpers ──────────────────────────────────────────────────────────

@always_inline
# ── Visibility with transmittance ─────────────────────────────────────────────

def _visible_transmittance(
    a: Point3f, b: Point3f,
    med_idx: Int32,
    ref sd:      SceneDescriptor2_C,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    wl:      SampledWavelengths,
) -> SpectralSample:
    """Returns transmittance along segment AB, spectrally, or black if
    occluded. Glass (dielectric) surfaces are passed through with Fresnel
    transmittance. `scratch` is one caller-owned Intersection_C slot (no
    internal alloc/free) so this is safe to call from a GPU kernel thread —
    every existing GPU kernel in this codebase takes pre-allocated,
    thread-indexed scratch instead of allocating per-thread (see
    sppm_gen_vp_gpu's inter_scratch).

    Each medium segment's sigma_t is upsampled to the 4 hero lanes FIRST
    (medium_sigma_t_spectral), then exponentiated PER LANE -- the same
    ordering fix as spectral_free_flight_weight, applied here to a plain
    deterministic Beer-Lambert evaluation rather than an importance-sampled
    ratio (no red-channel proposal to correct against; this just IS
    exp(-sigma_t(lambda)*d) at each segment, multiplied across segments,
    which is exact: exp(a)*exp(b) = exp(a+b))."""
    var d = b - a
    var dist_total = d.length()
    if dist_total < Float32(1e-5):
        return SpectralSample(Float32(0))
    var inv = Float32(1) / dist_total
    var dir = Vec3f(d.x*inv, d.y*inv, d.z*inv)

    var Tr = SpectralSample(Float32(1.0))
    var org = a + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
    var remaining = dist_total - Float32(0.0002)
    var cur_med = med_idx

    # Private local slot instead of the caller's `scratch`. The caller's slot
    # is simultaneously live in the enclosing traversal that called us, and
    # writing through both aliases is what the GPU build faults on.
    var _local_inter = InlineArray[Intersection_C, 1](fill=Intersection_C(
        PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var inter_mem = _local_inter.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for _ in range(8):
        if remaining < Float32(1e-4): break
        var ray = Ray_C(org, Vec3f(dir[0], dir[1], dir[2]))
        inter_mem[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, remaining * Float32(0.9995), inter_mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        # test_spheres (analytic spheres, e.g. the caustic sphere) aren't part of
        # the BVH — traverse_bvh2_core only tests triangles/curves — so they need
        # a separate pass. Seed a sentinel tHit=remaining*0.9995 when the BVH found
        # nothing, so test_spheres's own internal tMax (it only bounds itself by
        # result[0].tHit when result[0].hit is already set) respects the shadow
        # ray's segment length instead of defaulting to unbounded (1e38).
        var had_bvh_hit = inter_mem[unsafe_offset=0].hit != Int8(0)
        if not had_bvh_hit:
            inter_mem[unsafe_offset=0].hit = Int8(1)
            inter_mem[unsafe_offset=0].tHit = remaining * Float32(0.9995)
            # Clear the primId along with the sentinel. `scratch` is a
            # caller-owned slot reused across bounces, samples and (on GPU)
            # threads, so on a BVH miss the primId still holds STALE data
            # from a previous traversal -- and test_spheres writes none when
            # sphereCount == 0. The `primId.type != 4` test just below then
            # reads that stale type, and a leftover 4 makes a pure miss
            # masquerade as a sphere hit, falling through to
            # sd.spheres[stale id1] and sd.materials[stale materialIndex]:
            # an out-of-bounds read on any scene with no spheres.
            inter_mem[unsafe_offset=0].primId.type = Int8(0)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, inter_mem)
        if not had_bvh_hit and inter_mem[unsafe_offset=0].primId.type != Int8(4):
            inter_mem[unsafe_offset=0].hit = Int8(0)
        if inter_mem[unsafe_offset=0].hit == Int8(0):
            # Nothing between here and destination: apply remaining Beer-Lambert
            if Int(cur_med) >= 0:
                var med = sd.mediums[unsafe_offset=Int(cur_med)]
                var st_spec = medium_sigma_t_spectral(med, wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                Tr *= SpectralSample(exp(-st_spec.v0*remaining), exp(-st_spec.v1*remaining), exp(-st_spec.v2*remaining), exp(-st_spec.v3*remaining))
            break

        var inter = inter_mem[unsafe_offset=0]
        var t_hit = inter.tHit
        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = org + Vec3f(dir[0], dir[1], dir[2]) * t_hit

        # Beer-Lambert through medium segment up to hit
        if Int(cur_med) >= 0:
            var med = sd.mediums[unsafe_offset=Int(cur_med)]
            var st_spec = medium_sigma_t_spectral(med, wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
            Tr *= SpectralSample(exp(-st_spec.v0*t_hit), exp(-st_spec.v1*t_hit), exp(-st_spec.v2*t_hit), exp(-st_spec.v3*t_hit))

        if mat.type == MatKind.thin_dielectric or (
                mat.type == MatKind.dielectric and mat.sss_boundary != Int8(0)):
            # ... and a SUBSURFACE boundary, which is a dielectric too but
            # whose transport the BSSRDF models separately -- blocking it
            # costs sss-slab.vcm 48% of its energy (a pinned smoke cell),
            # so the straight-line crossing is load-bearing there.
            # A THIN dielectric only. Straight-line pass-through is valid
            # here because a thin slab's entry and exit refractions cancel --
            # the ray leaves parallel to how it arrived, so the shadow ray's
            # geometry is right and only the Fresnel attenuation is needed.
            #
            # A THICK dielectric used to pass through here too, and that was
            # wrong: light REFRACTS at a thick refractor, so a straight shot
            # through it is not a physical path at all. NEE was therefore
            # manufacturing transport that no sampling strategy can generate,
            # and MIS cannot cancel what it never sees. Measured on
            # barcelona-pavilion-day, whose pool is a water plane over a
            # coateddiffuse bottom: the bottom third of the frame read 2.243x
            # a pbrt BDPT reference with this pass-through and 1.461x without
            # -- by far the largest single error in that scene, and confined
            # to exactly the region where a shadow ray must cross the water.
            # The path tracer never had this bug: its shadow ray is a binary
            # any_hit test, so the water blocks it outright, which is
            # accidentally correct for a thick refractor. pbrt blocks it too.
            #
            # What legitimately DOES get through a thick refractor is the
            # bent path, and finding that is MNEE's job
            # (_bdpt_mnee_diffuse_area_light, which fires precisely for the
            # glass-obscured case) -- not this straight line.
            # Pass through glass with Fresnel transmittance
            var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            var facing = dot(dir, gn) < Float32(0)
            var n_for_cos = gn if facing else gn*Float32(-1)
            var cos_i = -dot(dir, n_for_cos)
            if cos_i < Float32(0): cos_i = -cos_i
            var ior = mat.albedo.r
            var fr = fr_dielectric(cos_i, Float32(1)/ior if facing else ior)
            var T = Float32(1) - fr
            Tr *= T
            if Tr.is_black():
                return SpectralSample(Float32(0))
            # Update medium after crossing glass surface
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[unsafe_offset=Int(mat.medium_interface_idx)]
                var md = dir[0]*gn[0]+dir[1]*gn[1]+dir[2]*gn[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        elif mat.type == MatKind.interface:
            # Pure medium boundary: update medium, continue
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[unsafe_offset=Int(mat.medium_interface_idx)]
                # An ANALYTIC SPHERE boundary has no mesh to read a normal
                # from -- _geom_normal would index sd.meshes with a sphere's
                # primId fields and return garbage, making the inside/outside
                # test below a coin flip. The dielectric branch above already
                # special-cases this; the interface branch did not, so roughly
                # half the shadow rays leaving a sphere-bounded medium kept
                # cur_med set to the medium and were then Beer-Lambert'd across
                # the vacuum outside it -- annihilating them. That halved every
                # volume NEE contribution at every scatter order (measured
                # 0.500x vs the path tracer on a sphere-bounded fog).
                var igna = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                var md = dir[0]*igna[0]+dir[1]*igna[1]+dir[2]*igna[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        else:
            # Opaque surface blocks the segment
            return SpectralSample(Float32(0))

    return Tr

@always_inline
def _bdpt_simple_light_count(ref sd: SceneDescriptor2_C) -> Int:
    """Number of lights reachable through _bdpt_sample_simple_light: distant +
    point + sphere, the three every BDPT material-loop samples the SAME way.
    Area and infinite are NOT covered -- area gets its own MNEE-capable
    sampling (_bdpt_mnee_diffuse_area_light/_bdpt_mnee_sphere_light) and
    infinite draws its own 2 pcg floats via _sample_infinite_light_nee, so
    both stay written out at their call sites. Mirrors shading.mojo's
    _nee_simple_light_count, but there is no shared struct between the two
    files' light contexts (ShadeContext vs SceneDescriptor2_C) to unify them
    on, hence the parallel definition rather than a genuinely shared one."""
    return Int(sd.distantLightCount) + Int(sd.pointLightCount) + Int(sd.sphereCount)


@always_inline
def _bdpt_sample_simple_light(
    ref sd: SceneDescriptor2_C, i: Int, hit_point: Vec3f, mut pcg: PCG32,
) -> LightSample:
    """The i-th distant/point/sphere light. Unlike shading.mojo's twin
    (_nee_sample_simple_light), this returns ONLY the LightSample -- BDPT's
    own occlusion primitive (_bdpt_nee_contribute, immediately below) tests
    the segment out to the exact `ls.dist` via _visible_transmittance, which
    is media-aware and needs no per-light-type tmax shrink the way
    shading.mojo's boolean any-hit test does. There is therefore no
    sampler/tmax PAIRING to get wrong here the way bba82627 did -- one fewer
    thing this duplication could silently break, not zero, since every call
    site still had to agree on the SAMPLER itself and its argument order.

    ORDER IS LOAD-BEARING: distant, then point, then sphere -- matching every
    existing call site in this file already. Of the three only SPHERE draws
    from `pcg`, so this is a pure loop collapse everywhere it's used, not a
    reordering; see this file's individual conversions for confirmation each
    call site's ORIGINAL order already matched this one exactly."""
    var nd = Int(sd.distantLightCount)
    var np_ = Int(sd.pointLightCount)
    if i < nd:
        var ls_d = _sample_distant_light_nee(sd.distantLights[unsafe_offset=i])
        return ls_d^
    if i < nd + np_:
        var ls_p = _sample_point_light_nee(sd.pointLights[unsafe_offset=i - nd], hit_point)
        return ls_p^
    var si = i - nd - np_
    var ls_s = _sample_sphere_light_nee(sd.spheres[unsafe_offset=si], Int(sd.sphereCount), hit_point, pcg)
    return ls_s^


@always_inline
def _bdpt_nee_contribute(
    beta: SpectralSample,
    w: SpectralSample,
    ls: LightSample,
    hit: Point3f,
    gn: Vec3f,
    cur_med_idx: Int32,
    ref sd: SceneDescriptor2_C,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    wl: SampledWavelengths,
    eps: Float32 = Float32(0.0001),
    two_sided: Bool = False,
) -> SpectralSample:
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
        return SpectralSample(Float32(0))
    # For a TWO-SIDED lobe the offset must follow the light direction:
    # +gn for a direction on the -gn side starts the ray inside the surface
    # it just left. Opt-in, NOT automatic -- MNEE connects THROUGH GLASS, so
    # a one-sided lobe CAN arrive here with cos < 0 and a non-black weight,
    # and flipping the offset there would push its ray to the wrong side.
    var side_eps = eps
    if two_sided and dot(gn, Vec3f(ls.wi[0], ls.wi[1], ls.wi[2])) < Float32(0):
        side_eps = -eps
    var shadow_org = hit + vec3f(gn) * side_eps
    var shadow_end = shadow_org + Vec3f(ls.wi[0], ls.wi[1], ls.wi[2]) * ls.dist
    var Tr = _visible_transmittance(shadow_org, shadow_end, cur_med_idx, sd, scratch, wl)
    if not Tr.is_black():
        return beta * w * Tr
    return SpectralSample(Float32(0))

def _bdpt_mnee_diffuse_area_light(
    ref sd: SceneDescriptor2_C, hit: Point3f, gn: Vec3f, eff_alb: RGB,
    beta: SpectralSample, mut pcg: PCG32, wl: SampledWavelengths,
    ior: Float32 = Float32(1.0),
) -> SpectralSample:
    """Real MNEE (manifold next-event estimation, task #161): for a diffuse
    (or coateddiffuse base-layer, see `ior` below) camera vertex, probe
    whether a straight line toward a randomly-picked area light first hits
    dielectric glass, and if so solve for the true refracted connection via
    Newton iteration -- reusing shading.mojo's _mnee_walk/_mnee_walk2, the
    exact technique the plain path tracer's own _nee_area_lights already
    uses. This is WHY the plain path tracer correctly lights
    barcelona-pavilion (night) while bdpt.mojo's VCM connect/merge cannot:
    dielectric bounces are never stored as LVC vertices (see
    project_vcm_stage2_mis_derivation memory), so a light behind glass is
    structurally invisible to connect/merge, and its tiny solid angle
    makes unassisted BSDF-sampling hit it by pure luck only.

    Deliberately scoped to ONLY the glass-detected case. An earlier attempt
    added plain straight-line NEE for ALL area lights (not just
    glass-obscured ones) and was reverted: it double-counted with the
    EXISTING connect/merge estimator on ordinary, unobstructed lights
    (confirmed via git-stash A/B on cornell-box, ~38% over pbrt's
    reference). MNEE only ever fires for paths connect/merge structurally
    cannot represent anyway (a delta dielectric bounce in the middle of the
    connecting path), so there is no matching double-count risk here --
    when the probe does NOT hit glass first, this returns black and
    connect/merge (already correct for that ordinary case) are untouched.

    `ior` (2026-07-13 follow-up): the caller's coat IOR, applied as a
    `(1 - Fresnel(cos_s_x0, ior))` transmittance factor on the returned
    weight, matching _nee_weight_coated_diffuse_base's own formula shape
    for ordinary (non-MNEE) coateddiffuse NEE. Defaults to 1.0 for plain
    diffuse callers -- fr_dielectric(_, 1.0) is exactly 0 (no index
    mismatch means no reflection), so `1 - 0 = 1` recovers the original
    unweighted diffuse behavior exactly, not an approximation. Still
    diffuse-family only -- no material in this codebase does MNEE for
    conductor/hair/measured today. Curve-shaped area lights are skipped
    (no well-defined surface tangents for a swept tube), same as
    shading.mojo."""
    var n_area = Int(sd.areaLightCount)
    if n_area <= 0:
        return SpectralSample(Float32(0))
    var li = Int(pcg.next_uint() % UInt32(n_area))
    var al = sd.areaLights[unsafe_offset=li]
    if al.kind == Int8(1):
        return SpectralSample(Float32(0))
    var lmesh = sd.meshes[unsafe_offset=Int(al.meshIdx)]
    var n_tris = Int(max(Int(al.n_tris), 1))
    var ti = Int(pcg.next_uint() % UInt32(n_tris))
    var lb = ti * 3
    var lv0 = Int(lmesh.vertexIndices[unsafe_offset=lb]); var lv1 = Int(lmesh.vertexIndices[unsafe_offset=lb+1]); var lv2 = Int(lmesh.vertexIndices[unsafe_offset=lb+2])
    var lp0 = Vec3f(lmesh.points[unsafe_offset=lv0*4], lmesh.points[unsafe_offset=lv0*4+1], lmesh.points[unsafe_offset=lv0*4+2])
    var lp1 = Vec3f(lmesh.points[unsafe_offset=lv1*4], lmesh.points[unsafe_offset=lv1*4+1], lmesh.points[unsafe_offset=lv1*4+2])
    var lp2 = Vec3f(lmesh.points[unsafe_offset=lv2*4], lmesh.points[unsafe_offset=lv2*4+1], lmesh.points[unsafe_offset=lv2*4+2])
    var ru1 = pcg.next_float(); var ru2 = pcg.next_float(); var sr1 = sqrt(ru1)
    var light_point = lp0*(Float32(1)-sr1) + lp1*(sr1*(Float32(1)-ru2)) + lp2*(sr1*ru2)
    var ldp_du = lp1 - lp0
    var ldp_dv = lp2 - lp0

    var hit_v = hit.to_simd()
    var to_light = light_point - hit_v
    var dist_sq = dot(to_light, to_light)
    if dist_sq < Float32(1e-8) or al.total_area <= Float32(0):
        return SpectralSample(Float32(0))
    var dist = sqrt(dist_sq)
    var shadow_dir = to_light * (Float32(1) / dist)

    var probe_org = hit_v + shadow_dir * Float32(0.0002)
    var probe_ray = Ray_C(Point3f(probe_org[0], probe_org[1], probe_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
    var probe_tmax = dist * Float32(0.9995)
    var dummy_prim = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection_C(dummy_prim, probe_tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var probe_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe_ray, probe_tmax, probe_store.unsafe_ptr(),
                       sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    var probe_inter = probe_store[0]
    if probe_inter.hit == Int8(0) or probe_inter.primId.type != Int8(0):
        return SpectralSample(Float32(0))
    var probe_mat = sd.materials[unsafe_offset=Int(probe_inter.primId.materialIndex)]
    if probe_mat.type != MatKind.dielectric and probe_mat.type != MatKind.thin_dielectric:
        return SpectralSample(Float32(0))

    var (pmesh, pv0, pv1, pv2, ptok) = _get_tri_verts(probe_inter, sd.meshes)
    if not ptok:
        return SpectralSample(Float32(0))
    var pp0 = Vec3f(pmesh.points[unsafe_offset=pv0*4], pmesh.points[unsafe_offset=pv0*4+1], pmesh.points[unsafe_offset=pv0*4+2])
    var pp1 = Vec3f(pmesh.points[unsafe_offset=pv1*4], pmesh.points[unsafe_offset=pv1*4+1], pmesh.points[unsafe_offset=pv1*4+2])
    var pp2 = Vec3f(pmesh.points[unsafe_offset=pv2*4], pmesh.points[unsafe_offset=pv2*4+1], pmesh.points[unsafe_offset=pv2*4+2])
    var pdp_du = pp1 - pp0
    var pdp_dv = pp2 - pp0
    var pgeo_n3 = cross(pdp_du, pdp_dv)
    var pgeo_n_len = sqrt(dot(pgeo_n3, pgeo_n3))
    if pgeo_n_len <= Float32(1e-10):
        return SpectralSample(Float32(0))
    var pgeo_n_raw = pgeo_n3 * (Float32(1) / pgeo_n_len)
    var ior1 = probe_mat.albedo.r
    var eta1 = ior1 if dot(pgeo_n_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior1)
    var pgeo_n = pgeo_n_raw
    if dot(pgeo_n, shadow_dir) > Float32(0):
        pgeo_n = -pgeo_n
    var pu = probe_inter.u; var pvb = probe_inter.v
    var x1_init = pp0*(Float32(1)-pu-pvb) + pp1*pu + pp2*pvb

    var pdf_sel = Float32(1) / Float32(n_area)  # uniform light+point pick, matches this file's own convention

    var probe2_t0 = probe_inter.tHit
    var probe2_rem = (dist - probe2_t0) * Float32(0.9995)
    var probe2_org = x1_init + shadow_dir * Float32(0.0005)
    var probe2_inter = dummy_inter
    if probe2_rem > Float32(0.001):
        var probe2_ray = Ray_C(Point3f(probe2_org[0], probe2_org[1], probe2_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var probe2_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe2_ray, probe2_rem, probe2_store.unsafe_ptr(),
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        probe2_inter = probe2_store[0]

    if probe2_inter.hit != Int8(0) and probe2_inter.primId.type == Int8(0):
        var probe2_mat = sd.materials[unsafe_offset=Int(probe2_inter.primId.materialIndex)]
        if probe2_mat.type == MatKind.dielectric or probe2_mat.type == MatKind.thin_dielectric:
            # --- 2-vertex MNEE ---
            var (p2mesh, p2v0, p2v1, p2v2, p2ok) = _get_tri_verts(probe2_inter, sd.meshes)
            if not p2ok:
                return SpectralSample(Float32(0))
            var p2p0 = Vec3f(p2mesh.points[unsafe_offset=p2v0*4], p2mesh.points[unsafe_offset=p2v0*4+1], p2mesh.points[unsafe_offset=p2v0*4+2])
            var p2p1 = Vec3f(p2mesh.points[unsafe_offset=p2v1*4], p2mesh.points[unsafe_offset=p2v1*4+1], p2mesh.points[unsafe_offset=p2v1*4+2])
            var p2p2 = Vec3f(p2mesh.points[unsafe_offset=p2v2*4], p2mesh.points[unsafe_offset=p2v2*4+1], p2mesh.points[unsafe_offset=p2v2*4+2])
            var pdp_du2 = p2p1 - p2p0; var pdp_dv2 = p2p2 - p2p0
            var pgeo_n3_2 = cross(pdp_du2, pdp_dv2)
            var pgeo_n_len2 = sqrt(dot(pgeo_n3_2, pgeo_n3_2))
            if pgeo_n_len2 <= Float32(1e-10):
                return SpectralSample(Float32(0))
            var pgeo_n2_raw = pgeo_n3_2 * (Float32(1) / pgeo_n_len2)
            var ior2 = probe2_mat.albedo.r
            var eta2 = ior2 if dot(pgeo_n2_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior2)
            var pgeo_n2 = pgeo_n2_raw
            if dot(pgeo_n2, shadow_dir) > Float32(0):
                pgeo_n2 = -pgeo_n2
            var pu2 = probe2_inter.u; var pvb2 = probe2_inter.v
            var x2_init = p2p0*(Float32(1)-pu2-pvb2) + p2p1*pu2 + p2p2*pvb2
            var (ok2, x1_f2, x2_f2, bsdf_prod, dx1_dxl2) = _mnee_walk2(
                hit_v, light_point,
                x1_init, pgeo_n, pdp_du, pdp_dv, eta1,
                x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2,
                ldp_du, ldp_dv)
            if not ok2:
                return SpectralSample(Float32(0))
            var wi2f = hit_v - x1_f2
            var wi2fl = sqrt(dot(wi2f, wi2f))
            if wi2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wi2fn = wi2f * (Float32(1) / wi2fl)
            var cos_s_x0 = dot(gn, -wi2fn)
            if cos_s_x0 <= Float32(0):
                return SpectralSample(Float32(0))
            var G2 = min(abs(dot(wi2fn, pgeo_n)) / (wi2fl*wi2fl) * dx1_dxl2, Float32(2))
            var pdf_area2 = pdf_sel / al.total_area
            var wo2f = light_point - x2_f2
            var wo2fl = sqrt(dot(wo2f, wo2f))
            if wo2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wo2fn = wo2f * (Float32(1) / wo2fl)
            var vis2_org = x2_f2 + wo2fn * Float32(0.001)
            var vis2_ray = Ray_C(Point3f(vis2_org[0], vis2_org[1], vis2_org[2]), Vec3f(wo2fn[0], wo2fn[1], wo2fn[2]))
            if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis2_ray, wo2fl * Float32(0.999),
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                                  sd.spheres, Int(sd.sphereCount)):
                return SpectralSample(Float32(0))
            var coat_t2 = Float32(1.0) - fr_dielectric(cos_s_x0, ior)
            var f_r = bxdf_eval_diffuse(eff_alb) * coat_t2
            return (beta
                * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
                * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, wl)
                * (cos_s_x0 * G2 * bsdf_prod / pdf_area2))
        return SpectralSample(Float32(0))
    else:
        # --- 1-vertex MNEE ---
        var (mnee_ok, x1_f, det_b, eta_f) = _mnee_walk(hit_v, light_point, x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        if not mnee_ok:
            return SpectralSample(Float32(0))
        var wi_f = hit_v - x1_f
        var wi_len2_f = dot(wi_f, wi_f)
        var wo_f = light_point - x1_f
        var wo_len2_f = dot(wo_f, wo_f)
        if wi_len2_f <= Float32(1e-8) or wo_len2_f <= Float32(1e-8):
            return SpectralSample(Float32(0))
        var wi_len_f = sqrt(wi_len2_f)
        var wo_len_f = sqrt(wo_len2_f)
        var wi_fn = wi_f * (Float32(1) / wi_len_f)
        var wo_fn = wo_f * (Float32(1) / wo_len_f)
        var cos_s_x0 = dot(gn, -wi_fn)
        if cos_s_x0 <= Float32(0):
            return SpectralSample(Float32(0))
        var H3_f = -(wi_fn + wo_fn * eta_f)
        var H_len2_f = dot(H3_f, H3_f)
        if H_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var H_len_f = sqrt(H_len2_f)
        var H_f = H3_f * (Float32(1) / H_len_f)
        var dp_du_dot_n = dot(pdp_du, pgeo_n)
        var s3_f = pdp_du - pgeo_n * dp_du_dot_n
        var s_len2_f = dot(s3_f, s3_f)
        if s_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var s_f = s3_f * (Float32(1) / sqrt(s_len2_f))
        var t_f = cross(pgeo_n, s_f)
        var ilo_l = eta_f / (H_len_f * wo_len_f)
        var dHdu_l = (ldp_du - wo_fn * dot(wo_fn, ldp_du)) * ilo_l
        var dHdv_l = (ldp_dv - wo_fn * dot(wo_fn, ldp_dv)) * ilo_l
        dHdu_l -= H_f * dot(dHdu_l, H_f); dHdu_l = -dHdu_l
        dHdv_l -= H_f * dot(dHdv_l, H_f); dHdv_l = -dHdv_l
        var dc00 = dot(dHdu_l, s_f); var dc01 = dot(dHdv_l, s_f)
        var dc10 = dot(dHdu_l, t_f); var dc11 = dot(dHdv_l, t_f)
        var det_dc = dc00*dc11 - dc01*dc10
        var dx1_dxl = abs(det_dc) / max(abs(det_b), Float32(1e-8))
        var dw0_dx1 = abs(dot(wi_fn, pgeo_n)) / wi_len2_f
        var G = min(dw0_dx1 * dx1_dxl, Float32(2))
        var cosNI = abs(dot(pgeo_n, wi_fn))
        var cosHI = abs(dot(H_f, wi_fn))
        var cosTM = abs(dot(pgeo_n, H_f))
        var F_r = fr_dielectric(cosNI, eta_f)
        var T_f = Float32(1) - F_r
        var bsdf_s = T_f * cosHI / max(cosNI * cosTM * cosTM, Float32(1e-6))
        var pdf_area_x2 = pdf_sel / al.total_area
        var vis_org = x1_f + wo_fn * Float32(0.001)
        var vis_ray = Ray_C(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
        if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis_ray, wo_len_f * Float32(0.999),
                              sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                              sd.spheres, Int(sd.sphereCount)):
            return SpectralSample(Float32(0))
        var coat_t1 = Float32(1.0) - fr_dielectric(cos_s_x0, ior)
        var f_r = bxdf_eval_diffuse(eff_alb) * coat_t1
        return (beta
                * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
                * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, wl)
                * (cos_s_x0 * G * bsdf_s / pdf_area_x2))


def _bdpt_mnee_sphere_light(
    ref sd: SceneDescriptor2_C, hit: Point3f, gn: Vec3f, eff_alb: RGB,
    beta: SpectralSample, mut pcg: PCG32, sph_idx: Int, n_spheres: Int,
    wl: SampledWavelengths, ior: Float32 = Float32(1.0),
) -> SpectralSample:
    """Real MNEE (task #161 follow-up, 2026-07-13) against an ANALYTIC
    SPHERE area light behind glass -- sibling to
    _bdpt_mnee_diffuse_area_light (mesh/triangle lights), which live in a
    completely separate list (sd.spheres, not sd.areaLights). Deliberate
    FULL, SELF-CONTAINED DUPLICATE of that function's probe+Newton-walk
    body (not a shared helper) -- see this section's own investigation
    notes below for why.

    _sample_sphere_light_nee's existing solid-angle/cone sampling (used
    for ORDINARY sphere NEE elsewhere in this file) can't be reused here
    -- it only returns a sampled DIRECTION and a solid-angle pdf w.r.t.
    the shading point, no actual surface point or light-side
    parameterization to take Newton-walk derivatives against. Instead,
    samples a UNIFORM point on the sphere's surface via the standard
    spherical parameterization p(θ,φ) = center + r·(sinθcosφ, sinθsinφ,
    cosθ), using that SAME parameterization's own analytic partial
    derivatives ∂p/∂φ, ∂p/∂θ as the light-side tangent vectors (matches
    pbrt's own dpdu/dpdv convention for spheres).

    `sph_idx` (an index into sd.spheres, read locally via
    `sd.spheres[sph_idx]`) + `n_spheres` (the TOTAL sphere count, matching
    `_sample_sphere_light_nee`'s own `1/n_sph` pdf convention) -- caller
    iterates every sphere (see call-site comment for why NOT via a `for`
    loop). `ior`: accepted for signature symmetry with
    _bdpt_mnee_diffuse_area_light but NOT applied in the return value --
    see the GPU-codegen-bug note below for why.

    RESOLVED GPU BUG (2026-07-13, real Mojo/GPU-codegen bug, not sphere-
    specific -- root-caused via systematic bisection, not application
    logic): every earlier implementation of this feature crashed with a
    reproducible CUDA_ERROR_ILLEGAL_ADDRESS on barcelona-pavilion-night
    (this task's actual target scene). Made fully deterministic by
    temporarily hardcoding pbrt_parser.mojo's RNG seed (normally
    perf_counter_ns()) -- this turned a seemingly-nondeterministic crash
    (varied run to run because a wall-clock seed explores different pixel/
    sample paths each time) into 100%-reproducible pass/fail, which is
    what made real bisection possible. Systematic cutoff-return bisection
    through this function's body (return RGB(0) at successively later
    points, rebuild+rerun at each cutoff) narrowed the crash to an exact
    line: folding a SECOND `fr_dielectric(...)` call's result (`coat_t =
    1 - fr_dielectric(cos_s_x0, ior)`) into the final returned RGB
    expression, alongside `sph.emission`/`bxdf_eval_diffuse(...)`/the G
    and pdf terms. Calling `fr_dielectric` and discarding the result was
    SAFE; using `sph.emission` and `bxdf_eval_diffuse(...)` together in
    the final expression was SAFE; splitting the multiply across two
    statements (`var contrib = ...; return contrib * coat_t`) did NOT
    help -- still crashed identically, ruling out "too many chained
    multiplies in one expression" as the mechanism. This is consistent
    with the general shape of the anomaly logged in
    `reference_mojo_compiler_bug_6759.md` (heavy, `InlineArray`-using,
    multiple early returns, BVH traversal, under this scene's specific
    complexity) -- though that report was later retracted by its own
    author as unreproducible, so this crash stands on its own bisection
    below, not on 6759 as corroboration. Not a NaN/degenerate-value bug
    in this code (cos_s_x0/ior were always finite, well-conditioned
    values at the crash site).
    WORKAROUND (applied here): don't compute/apply `coat_t` at all. Every
    CURRENT call site passes the default `ior=1.0`, for which
    `fr_dielectric(_, 1.0) == 0` exactly (an identity already relied on
    by _bdpt_mnee_diffuse_area_light's own `ior=1.0` default case), so
    `coat_t` would always equal exactly `1.0` anyway -- omitting it is a
    zero behavior change today, not an approximation. If sphere-light
    MNEE for coateddiffuse (ior != 1.0) is ever revisited, `coat_t` will
    need a DIFFERENT strategy that avoids this exact pattern (e.g.
    precomputing it in the caller and passing it in as a parameter,
    rather than computing+applying `fr_dielectric` inside this function).
    See project_barcelona_pavilion_mnee memory for the full investigation,
    including two earlier, unrelated red herrings (a `for`-loop-wrapping
    hypothesis and a shared-function-split hypothesis, both ruled out by
    this same bisection).
    WORKAROUND (unrelated, applied at every call site): call this
    function via manually UNROLLED `if`-guarded statements, never a `for`
    loop, capped at a small constant (comptime _MNEE_MAX_SPHERES) --
    scenes with more emissive spheres than the cap silently skip the
    extras for MNEE only (ordinary light-hit/connect/merge/NEE still
    reaches them normally, this only affects the glass-behind-sphere-
    light special case). Kept even though the loop-wrapping hypothesis
    turned out not to be the real bug, since unrolling is harmless and
    was already in place before the real cause was found."""
    var sph = sd.spheres[unsafe_offset=sph_idx]
    if sph.isAreaLight == Int8(0):
        return SpectralSample(Float32(0))
    var u1 = pcg.next_float(); var u2 = pcg.next_float()
    var cosT = Float32(1) - Float32(2) * u1
    var sinT = sqrt(max(Float32(0), Float32(1) - cosT*cosT))
    var phi = Float32(2) * PI * u2
    var cosPhi = cos(phi); var sinPhi = sin(phi)
    var dir = Vec3f(sinT*cosPhi, sinT*sinPhi, cosT)
    var light_point = sph.center.to_simd() + dir * sph.radius
    # Analytic tangents of p(theta,phi) = center + r*dir(theta,phi) w.r.t.
    # (phi,theta) -- the same two parameters this point was just sampled
    # from (matches pbrt's own dpdu/dpdv convention for spheres).
    var ldp_du = Vec3f(-sinT*sinPhi, sinT*cosPhi, Float32(0)) * sph.radius   # dp/dphi
    var ldp_dv = Vec3f(cosT*cosPhi, cosT*sinPhi, -sinT) * sph.radius          # dp/dtheta
    var total_area = Float32(4) * PI * sph.radius * sph.radius
    if total_area <= Float32(0):
        return SpectralSample(Float32(0))
    var hit_v = hit.to_simd()
    var to_light = light_point - hit_v
    var dist_sq = dot(to_light, to_light)
    if dist_sq < Float32(1e-8) or total_area <= Float32(0):
        return SpectralSample(Float32(0))
    var dist = sqrt(dist_sq)
    var shadow_dir = to_light * (Float32(1) / dist)

    var probe_org = hit_v + shadow_dir * Float32(0.0002)
    var probe_ray = Ray_C(Point3f(probe_org[0], probe_org[1], probe_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
    var probe_tmax = dist * Float32(0.9995)
    var dummy_prim = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection_C(dummy_prim, probe_tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var probe_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe_ray, probe_tmax, probe_store.unsafe_ptr(),
                       sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    var probe_inter = probe_store[0]
    if probe_inter.hit == Int8(0) or probe_inter.primId.type != Int8(0):
        return SpectralSample(Float32(0))
    var probe_mat = sd.materials[unsafe_offset=Int(probe_inter.primId.materialIndex)]
    if probe_mat.type != MatKind.dielectric and probe_mat.type != MatKind.thin_dielectric:
        return SpectralSample(Float32(0))

    var (pmesh, pv0, pv1, pv2, ptok) = _get_tri_verts(probe_inter, sd.meshes)
    if not ptok:
        return SpectralSample(Float32(0))
    var pp0 = Vec3f(pmesh.points[unsafe_offset=pv0*4], pmesh.points[unsafe_offset=pv0*4+1], pmesh.points[unsafe_offset=pv0*4+2])
    var pp1 = Vec3f(pmesh.points[unsafe_offset=pv1*4], pmesh.points[unsafe_offset=pv1*4+1], pmesh.points[unsafe_offset=pv1*4+2])
    var pp2 = Vec3f(pmesh.points[unsafe_offset=pv2*4], pmesh.points[unsafe_offset=pv2*4+1], pmesh.points[unsafe_offset=pv2*4+2])
    var pdp_du = pp1 - pp0
    var pdp_dv = pp2 - pp0
    var pgeo_n3 = cross(pdp_du, pdp_dv)
    var pgeo_n_len = sqrt(dot(pgeo_n3, pgeo_n3))
    if pgeo_n_len <= Float32(1e-10):
        return SpectralSample(Float32(0))
    var pgeo_n_raw = pgeo_n3 * (Float32(1) / pgeo_n_len)
    var ior1 = probe_mat.albedo.r
    var eta1 = ior1 if dot(pgeo_n_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior1)
    var pgeo_n = pgeo_n_raw
    if dot(pgeo_n, shadow_dir) > Float32(0):
        pgeo_n = -pgeo_n
    var pu = probe_inter.u; var pvb = probe_inter.v
    var x1_init = pp0*(Float32(1)-pu-pvb) + pp1*pu + pp2*pvb

    var pdf_sel = Float32(1) / Float32(max(n_spheres, 1))  # uniform light+point pick, matches this file's own convention

    var probe2_t0 = probe_inter.tHit
    var probe2_rem = (dist - probe2_t0) * Float32(0.9995)
    var probe2_org = x1_init + shadow_dir * Float32(0.0005)
    var probe2_inter = dummy_inter
    if probe2_rem > Float32(0.001):
        var probe2_ray = Ray_C(Point3f(probe2_org[0], probe2_org[1], probe2_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var probe2_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe2_ray, probe2_rem, probe2_store.unsafe_ptr(),
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        probe2_inter = probe2_store[0]

    if probe2_inter.hit != Int8(0) and probe2_inter.primId.type == Int8(0):
        var probe2_mat = sd.materials[unsafe_offset=Int(probe2_inter.primId.materialIndex)]
        if probe2_mat.type == MatKind.dielectric or probe2_mat.type == MatKind.thin_dielectric:
            # --- 2-vertex MNEE ---
            var (p2mesh, p2v0, p2v1, p2v2, p2ok) = _get_tri_verts(probe2_inter, sd.meshes)
            if not p2ok:
                return SpectralSample(Float32(0))
            var p2p0 = Vec3f(p2mesh.points[unsafe_offset=p2v0*4], p2mesh.points[unsafe_offset=p2v0*4+1], p2mesh.points[unsafe_offset=p2v0*4+2])
            var p2p1 = Vec3f(p2mesh.points[unsafe_offset=p2v1*4], p2mesh.points[unsafe_offset=p2v1*4+1], p2mesh.points[unsafe_offset=p2v1*4+2])
            var p2p2 = Vec3f(p2mesh.points[unsafe_offset=p2v2*4], p2mesh.points[unsafe_offset=p2v2*4+1], p2mesh.points[unsafe_offset=p2v2*4+2])
            var pdp_du2 = p2p1 - p2p0; var pdp_dv2 = p2p2 - p2p0
            var pgeo_n3_2 = cross(pdp_du2, pdp_dv2)
            var pgeo_n_len2 = sqrt(dot(pgeo_n3_2, pgeo_n3_2))
            if pgeo_n_len2 <= Float32(1e-10):
                return SpectralSample(Float32(0))
            var pgeo_n2_raw = pgeo_n3_2 * (Float32(1) / pgeo_n_len2)
            var ior2 = probe2_mat.albedo.r
            var eta2 = ior2 if dot(pgeo_n2_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior2)
            var pgeo_n2 = pgeo_n2_raw
            if dot(pgeo_n2, shadow_dir) > Float32(0):
                pgeo_n2 = -pgeo_n2
            var pu2 = probe2_inter.u; var pvb2 = probe2_inter.v
            var x2_init = p2p0*(Float32(1)-pu2-pvb2) + p2p1*pu2 + p2p2*pvb2
            var (ok2, x1_f2, x2_f2, bsdf_prod, dx1_dxl2) = _mnee_walk2(
                hit_v, light_point,
                x1_init, pgeo_n, pdp_du, pdp_dv, eta1,
                x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2,
                ldp_du, ldp_dv)
            if not ok2:
                return SpectralSample(Float32(0))
            var wi2f = hit_v - x1_f2
            var wi2fl = sqrt(dot(wi2f, wi2f))
            if wi2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wi2fn = wi2f * (Float32(1) / wi2fl)
            var cos_s_x0 = dot(gn, -wi2fn)
            if cos_s_x0 <= Float32(0):
                return SpectralSample(Float32(0))
            var G2 = min(abs(dot(wi2fn, pgeo_n)) / (wi2fl*wi2fl) * dx1_dxl2, Float32(2))
            var pdf_area2 = pdf_sel / total_area
            var wo2f = light_point - x2_f2
            var wo2fl = sqrt(dot(wo2f, wo2f))
            if wo2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wo2fn = wo2f * (Float32(1) / wo2fl)
            var vis2_org = x2_f2 + wo2fn * Float32(0.001)
            var vis2_ray = Ray_C(Point3f(vis2_org[0], vis2_org[1], vis2_org[2]), Vec3f(wo2fn[0], wo2fn[1], wo2fn[2]))
            if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis2_ray, wo2fl * Float32(0.999),
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                                  sd.spheres, Int(sd.sphereCount)):
                return SpectralSample(Float32(0))
            # coat_t (coateddiffuse coat-transmittance, mirrors
            # _bdpt_mnee_diffuse_area_light's `ior` handling) is DELIBERATELY
            # not applied here -- see this function's own docstring, "GPU
            # codegen bug" section, for why folding a 2nd fr_dielectric(...)
            # result into this return crashes with CUDA_ERROR_ILLEGAL_ADDRESS
            # on this task's target scene. Every current call site passes
            # the default ior=1.0 (coateddiffuse sphere-light call sites are
            # disabled, see _MNEE_MAX_SPHERES call-site comments), for which
            # fr_dielectric(_, 1.0) == 0 exactly, so coat_t == 1.0 exactly --
            # applying it would be a mathematical no-op anyway.
            var f_r = bxdf_eval_diffuse(eff_alb)
            return (beta
                * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
                * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sph.emission.r, sph.emission.g, sph.emission.b, wl)
                * (cos_s_x0 * G2 * bsdf_prod / pdf_area2))
        return SpectralSample(Float32(0))
    else:
        # --- 1-vertex MNEE ---
        var (mnee_ok, x1_f, det_b, eta_f) = _mnee_walk(hit_v, light_point, x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        if not mnee_ok:
            return SpectralSample(Float32(0))
        var wi_f = hit_v - x1_f
        var wi_len2_f = dot(wi_f, wi_f)
        var wo_f = light_point - x1_f
        var wo_len2_f = dot(wo_f, wo_f)
        if wi_len2_f <= Float32(1e-8) or wo_len2_f <= Float32(1e-8):
            return SpectralSample(Float32(0))
        var wi_len_f = sqrt(wi_len2_f)
        var wo_len_f = sqrt(wo_len2_f)
        var wi_fn = wi_f * (Float32(1) / wi_len_f)
        var wo_fn = wo_f * (Float32(1) / wo_len_f)
        var cos_s_x0 = dot(gn, -wi_fn)
        if cos_s_x0 <= Float32(0):
            return SpectralSample(Float32(0))
        var H3_f = -(wi_fn + wo_fn * eta_f)
        var H_len2_f = dot(H3_f, H3_f)
        if H_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var H_len_f = sqrt(H_len2_f)
        var H_f = H3_f * (Float32(1) / H_len_f)
        var dp_du_dot_n = dot(pdp_du, pgeo_n)
        var s3_f = pdp_du - pgeo_n * dp_du_dot_n
        var s_len2_f = dot(s3_f, s3_f)
        if s_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var s_f = s3_f * (Float32(1) / sqrt(s_len2_f))
        var t_f = cross(pgeo_n, s_f)
        var ilo_l = eta_f / (H_len_f * wo_len_f)
        var dHdu_l = (ldp_du - wo_fn * dot(wo_fn, ldp_du)) * ilo_l
        var dHdv_l = (ldp_dv - wo_fn * dot(wo_fn, ldp_dv)) * ilo_l
        dHdu_l -= H_f * dot(dHdu_l, H_f); dHdu_l = -dHdu_l
        dHdv_l -= H_f * dot(dHdv_l, H_f); dHdv_l = -dHdv_l
        var dc00 = dot(dHdu_l, s_f); var dc01 = dot(dHdv_l, s_f)
        var dc10 = dot(dHdu_l, t_f); var dc11 = dot(dHdv_l, t_f)
        var det_dc = dc00*dc11 - dc01*dc10
        var dx1_dxl = abs(det_dc) / max(abs(det_b), Float32(1e-8))
        var dw0_dx1 = abs(dot(wi_fn, pgeo_n)) / wi_len2_f
        var G = min(dw0_dx1 * dx1_dxl, Float32(2))
        var cosNI = abs(dot(pgeo_n, wi_fn))
        var cosHI = abs(dot(H_f, wi_fn))
        var cosTM = abs(dot(pgeo_n, H_f))
        var F_r = fr_dielectric(cosNI, eta_f)
        var T_f = Float32(1) - F_r
        var bsdf_s = T_f * cosHI / max(cosNI * cosTM * cosTM, Float32(1e-6))
        var pdf_area_x2 = pdf_sel / total_area
        var vis_org = x1_f + wo_fn * Float32(0.001)
        var vis_ray = Ray_C(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
        if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis_ray, wo_len_f * Float32(0.999),
                              sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                              sd.spheres, Int(sd.sphereCount)):
            return SpectralSample(Float32(0))
        # coat_t deliberately not applied -- see the 2-vertex branch's
        # identical comment above (this function's docstring has the full
        # GPU-codegen-bug writeup). ior=1.0 at every current call site makes
        # this an exact no-op, not an approximation.
        var f_r = bxdf_eval_diffuse(eff_alb)
        return (beta
            * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
            * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sph.emission.r, sph.emission.g, sph.emission.b, wl)
            * (cos_s_x0 * G * bsdf_s / pdf_area_x2))

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
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
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
    lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS + local_idx] = v

# ── LVC connection scale factor ──────────────────────────────────────────────

@always_inline
@always_inline
def _bdpt_world_to_raster(
    p_world: Vec3f,
    w2c: Pointer[Float32, MutUntrackedOrigin],     # inverse(cameraToWorld), col-major
    c2r: Pointer[Float32, MutUntrackedOrigin],     # inverse of the 3x3 raster->camera map, row-major
    fw: Int32, fh: Int32,
) -> Tuple[Bool, Float32, Float32, Float32]:
    """Project a world point onto the film. Returns (ok, filmX, filmY,
    cos_theta), where cos_theta is the angle between the camera's forward
    axis and the direction to the point.

    This is the inverse of sampling.mojo's `gen_primary_ray_state` camera
    transform, which builds a ray direction as `M . (filmX, filmY, 1)` with
    M the 3x3 taken from rasterToCamera's columns 0, 1 and 3 (its z column
    is unused because the film sits at z=0). Inverting that map and
    dividing through by the third component recovers (filmX, filmY) for any
    camera-space direction, which is exactly what the t=1 light-tracing
    strategy needs and what nothing in this renderer could do before.

    ok=False when the point is behind the camera or lands off-film."""
    # world -> camera
    var px = p_world[0]; var py = p_world[1]; var pz = p_world[2]
    var cx = w2c[unsafe_offset=0]*px + w2c[unsafe_offset=4]*py + w2c[unsafe_offset=8]*pz  + w2c[unsafe_offset=12]
    var cy = w2c[unsafe_offset=1]*px + w2c[unsafe_offset=5]*py + w2c[unsafe_offset=9]*pz  + w2c[unsafe_offset=13]
    var cz = w2c[unsafe_offset=2]*px + w2c[unsafe_offset=6]*py + w2c[unsafe_offset=10]*pz + w2c[unsafe_offset=14]
    if cz <= Float32(1e-6):
        return (False, Float32(0), Float32(0), Float32(0))   # behind the lens
    var clen = sqrt(cx*cx + cy*cy + cz*cz)
    if clen <= Float32(1e-12):
        return (False, Float32(0), Float32(0), Float32(0))
    var cos_theta = cz / clen                                # forward axis is +z
    # camera-space direction -> (filmX, filmY): q = C2R . c, then divide
    var q0 = c2r[unsafe_offset=0]*cx + c2r[unsafe_offset=1]*cy + c2r[unsafe_offset=2]*cz
    var q1 = c2r[unsafe_offset=3]*cx + c2r[unsafe_offset=4]*cy + c2r[unsafe_offset=5]*cz
    var q2 = c2r[unsafe_offset=6]*cx + c2r[unsafe_offset=7]*cy + c2r[unsafe_offset=8]*cz
    if abs(q2) <= Float32(1e-12):
        return (False, Float32(0), Float32(0), Float32(0))
    var fx = q0 / q2
    var fy = q1 / q2
    if fx < Float32(0) or fy < Float32(0) or fx >= Float32(fw) or fy >= Float32(fh):
        return (False, Float32(0), Float32(0), Float32(0))
    return (True, fx, fy, cos_theta)

@always_inline
def _bdpt_splat_filtered[use_atomics: Bool](
    accum: Pointer[Float32, MutUntrackedOrigin],   # 3 floats per pixel
    fx: Float32, fy: Float32,                     # continuous raster position
    rgb_r: Float32, rgb_g: Float32, rgb_b: Float32,
    fw: Int, fh: Int,
    ff: FilmFilter,
):
    """Spread one t=1 splat over every pixel its PixelFilter footprint covers,
    weighted f(p - pixel centre) / integral(f) -- pbrt-v4's RGBFilm::AddSplat,
    which normalises the splat by the filter integral at output time.

    Splats used to land on the single pixel containing p. That is a box
    filter, so while the camera half of VCM now reconstructs through the
    scene's PixelFilter, its light-traced half would have stayed box-sharp --
    the two halves of one image filtered differently. For a half-pixel box this
    is exactly the old behaviour: weight 1, one pixel. Energy is conserved in
    expectation, since the weights over the footprint sum to integral(f)."""
    var ftype = ff[0].cast[DType.int32]()
    var rx = ff[2]
    var ry = ff[3]
    var inv_int = Float32(1.0) / max(filter_integral_2d(ftype, ff[1], rx, ry), Float32(1e-12))
    var x0 = max(0, Int(floor(fx + Float32(0.5) - rx)))
    var x1 = min(fw - 1, Int(floor(fx + Float32(0.5) + rx)))
    var y0 = max(0, Int(floor(fy + Float32(0.5) - ry)))
    var y1 = min(fh - 1, Int(floor(fy + Float32(0.5) + ry)))
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            var w = filter_eval_2d(fx - (Float32(px) + Float32(0.5)), fy - (Float32(py) + Float32(0.5)),
                                   ftype, ff[1], rx, ry) * inv_int
            if w <= Float32(0.0):
                continue
            var o = (py * fw + px) * 3
            comptime if use_atomics:
                _ = Atomic[DType.float32].fetch_add(accum.unsafe_offset(o + 0), rgb_r * w)
                _ = Atomic[DType.float32].fetch_add(accum.unsafe_offset(o + 1), rgb_g * w)
                _ = Atomic[DType.float32].fetch_add(accum.unsafe_offset(o + 2), rgb_b * w)
            else:
                accum[unsafe_offset=o + 0] += rgb_r * w
                accum[unsafe_offset=o + 1] += rgb_g * w
                accum[unsafe_offset=o + 2] += rgb_b * w

def _bdpt_connect_to_camera(
    lv: BDPTVertex,
    ref sd: SceneDescriptor2_C,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    cam_pos: Vec3f,
    w2c: Pointer[Float32, MutUntrackedOrigin],
    c2r: Pointer[Float32, MutUntrackedOrigin],
    fw: Int32, fh: Int32,
    px_scale: Float32,
    n_light_paths_f: Float32,
    mis_vm_weight_factor: Float32,
) -> Tuple[Bool, Float32, Float32, SpectralSample]:
    """The t=1 strategy: connect a LIGHT-subpath vertex directly to the
    camera and return the continuous raster position it lands on plus its
    contribution. The caller spreads it over the filter footprint
    (_bdpt_splat_filtered), as pbrt's RGBFilm::AddSplat does.
    Ported from SmallVCM's `ConnectToCamera` (vertexcm.hxx), the same
    reference the rest of this file's MIS quantities came from.

    This strategy did not exist here before, and measurement says it is
    20.8% of a cornell-box image (pbrt's own per-strategy BDPT output).
    Since `_connect`'s MIS weight already RESERVES its share through the
    dVC/dVCM recursion, leaving it out did not merely omit those paths --
    it under-weighted every other strategy by that share, which is what
    made --vcm come out at 0.774 of gonzales's own path tracer.

    Radiometry, following SmallVCM exactly:

        imagePlaneDist       = 1 / px_scale        (pixels per world unit
                                                    at unit distance -- the
                                                    same convention the
                                                    camera-path dVCM's
                                                    cameraPdfW already uses)
        imageToSolidAngle    = (imagePlaneDist/cosAtCamera)^2 / cosAtCamera
        imageToSurface       = imageToSolidAngle * |cosToCamera| / dist^2
        contrib              = beta * f * imageToSurface / lightSubPathCount

    with one deliberate difference in bookkeeping: SmallVCM's
    `BSDF::Evaluate` returns the BSDF WITHOUT the outgoing cosine and picks
    it up again inside imageToSurface, whereas this file's `_eval_vertex`
    returns BSDF x cos. So the cosine is taken from `_eval_vertex` and
    imageToSolidAngle is used here WITHOUT it -- multiplying both would
    square the cosine and darken grazing geometry.

    Only MIS-scoped vertices are connected, and never the light-source
    vertex (see the gate below for why that one is excluded). An unscoped
    vertex gets `weight = 1` from `_connect`, which that function documents
    as already being the complete estimate for its kind; splatting such a
    vertex too would double-count it.

    Returns (ok, pixel_index, contribution)."""
    if lv.is_delta != Int32(0) or lv.is_surface == Int32(0):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    # The light-source vertex itself is NEVER splatted. Connecting the
    # emission point straight to the camera builds the length-1 path
    # "camera sees the emitter", which the camera path already credits in
    # full: its primary ray starts with last_bsdf_pdf = -1, so a direct hit on
    # an area light takes weight 1. Both claimed the whole emitter -- every
    # directly visible light rendered at exactly 2x (cornell-box's emitter
    # pixels read 1.978x the path tracer while the rest of the image matched).
    #
    # SmallVCM makes the same choice for the same reason: its light paths
    # connect to the camera only from their first BOUNCE onward, which is
    # precisely why its GetLightRadiance can return weight 1 at path length 1.
    # This keeps that pairing instead of re-weighting the camera hit, and it
    # is also the lower-variance estimator of the two -- every pixel that sees
    # the emitter sees it on its own camera ray, with no noise from where
    # light paths happened to start.
    #
    # lv0 stays in the light-vertex cache: CAMERA-VERTEX connections to it are
    # the s=1 strategy at path length >= 2, a different thing entirely.
    if lv.is_light == Int32(1) or not _bdpt_vertex_mis_scoped(lv):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var lp = lv.pos.to_simd()
    var d3 = cam_pos - lp
    var dist2 = dot(d3, d3)
    if dist2 < Float32(1e-8):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    var dist = sqrt(dist2)
    var dir_to_cam = d3 * (Float32(1) / dist)

    var pr = _bdpt_world_to_raster(lp, w2c, c2r, fw, fh)
    if not pr[0]:
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    var cos_at_camera = pr[3]
    if cos_at_camera <= Float32(1e-6):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var wl = lv.wavelengths
    var cos_to_camera = abs(dot(lv.normal.to_simd(), dir_to_cam))
    var f = _eval_vertex_spectral(lv, Vec3f(dir_to_cam[0], dir_to_cam[1], dir_to_cam[2]), sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
    if f.is_black():
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    if cos_to_camera <= Float32(1e-8):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var Tr = _visible_transmittance(
        lv.pos, Point3f(cam_pos[0], cam_pos[1], cam_pos[2]), lv.med_idx, sd, scratch, wl)
    if Tr.is_black():
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var image_plane_dist = Float32(1) / max(px_scale, Float32(1e-12))
    var ipcd = image_plane_dist / cos_at_camera
    var image_to_solid_angle = (ipcd * ipcd) / cos_at_camera
    # `_eval_vertex` already carries |cos| at the light vertex, so it is
    # deliberately NOT reapplied here (see the docstring).
    var geom = image_to_solid_angle / dist2
    var image_to_surface = image_to_solid_angle * cos_to_camera / dist2

    var inv_n = Float32(1) / max(n_light_paths_f, Float32(1))
    var contrib = (lv.beta * f * Tr
                   * (geom * inv_n))

    # MIS, SmallVCM ConnectToCamera:
    #   wLight = (cameraPdfA / lightSubPathCount)
    #            * (misVmWeightFactor + dVCM + dVC * bsdfRevPdfW)
    #   weight = 1 / (wLight + 1)
    var (_dp, rev_pdf_w) = _bdpt_vertex_pdfs(lv, Vec3f(dir_to_cam[0], dir_to_cam[1], dir_to_cam[2]), sd)
    var camera_pdf_a = image_to_surface
    var w_light = (camera_pdf_a * inv_n) * (mis_vm_weight_factor + lv.dVCM + lv.dVC * rev_pdf_w)
    var mis_weight = Float32(1) / (w_light + Float32(1))
    contrib = contrib * mis_weight

    return (True, pr[1], pr[2], contrib)

def _bdpt_connect_to_cache(
    cv: BDPTVertex,
    ref sd: SceneDescriptor2_C,
    has_med: Bool,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx: Int,
    path_len: Int,
    mis_vm_weight_factor: Float32,
) -> SpectralSample:
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
    var sum = SpectralSample(Float32(0))
    # Sum every MIS-WEIGHTED pair, but take at most ONE UNWEIGHTED pair.
    # Volume vertices have no dVCM/dVC (no surface pdf), so any (s,t) split
    # touching one comes back unweighted -- summing every such split
    # over-counts (see docs/09_volumetric_media.md, "VCM/BDPT volume
    # connections": the 1.027/1.388/1.651/1.969x non-decaying-increment
    # signature). Keeping only the lowest-index unweighted pair restores
    # exactly one strategy per path length, unbiased. Keyed on the PAIR, not
    # on which side is in a medium -- a surface cv into several volume light
    # vertices is the same over-count from the other side.
    var took_unweighted = False
    for local in range(path_len):
        var lv = lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS + local]
        if not _bdpt_connect_pair_weighted(cv, lv):
            if took_unweighted:
                continue
            took_unweighted = True
        sum += _connect(cv, lv, sd, has_med, scratch, mis_vm_weight_factor)
    return sum

def _bdpt_connect_to_cache_deferred(
    cv: BDPTVertex,
    ref sd: SceneDescriptor2_C,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx: Int,
    path_len: Int,
    mis_vm_weight_factor: Float32,
    shadow_rays: Pointer[Float32, MutUntrackedOrigin],
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin],
):
    """Task #163 stage 5: Vulkan-RT-batched counterpart to
    _bdpt_connect_to_cache -- instead of resolving each connection's shadow
    ray inline via software BVH (_visible_transmittance), writes the ray +
    _connect_unweighted's unweighted contribution into `lp_idx`'s own
    dedicated _BDPT_MAX_VERTS-sized slice of the shadow-ray queue buffers
    (same per-path-slice indexing convention as the LVC itself, see
    _bdpt_store_lvc_vertex's docstring), for a later batched Vulkan RT
    dispatch + resolve pass to fill in. ALWAYS writes exactly
    _BDPT_MAX_VERTS slots, marking unused/invalid ones so stale data from a
    previous bounce or sample never leaks through a reused buffer."""
    var base = lp_idx * _BDPT_MAX_VERTS
    # Same "every weighted pair, at most one unweighted pair" rule as
    # _bdpt_connect_to_cache -- see its comment for the derivation and the
    # measurement. Slots skipped by the rule are marked invalid, exactly like
    # slots past path_len, so no stale contribution leaks through.
    var took_unweighted = False
    for local in range(_BDPT_MAX_VERTS):
        if local >= path_len:
            shadow_valid[unsafe_offset=base + local] = Int8(0)
            continue
        var lv = lvc[unsafe_offset=base + local]
        if not _bdpt_connect_pair_weighted(cv, lv):
            if took_unweighted:
                shadow_valid[unsafe_offset=base + local] = Int8(0)
                continue
            took_unweighted = True
        var (contrib, valid) = _connect_unweighted(cv, lv, sd, mis_vm_weight_factor)
        if not valid:
            shadow_valid[unsafe_offset=base + local] = Int8(0)
            continue
        var d3 = lv.pos - cv.pos
        var dist = sqrt(d3.length_sq())
        var dir = d3.to_simd() / dist
        var idx8 = (base + local) * 8
        shadow_rays[unsafe_offset=idx8 + 0] = cv.pos.x
        shadow_rays[unsafe_offset=idx8 + 1] = cv.pos.y
        shadow_rays[unsafe_offset=idx8 + 2] = cv.pos.z
        shadow_rays[unsafe_offset=idx8 + 3] = Float32(1e-4)
        shadow_rays[unsafe_offset=idx8 + 4] = dir[0]
        shadow_rays[unsafe_offset=idx8 + 5] = dir[1]
        shadow_rays[unsafe_offset=idx8 + 6] = dir[2]
        shadow_rays[unsafe_offset=idx8 + 7] = dist * Float32(0.9995)
        shadow_pending[unsafe_offset=base + local] = contrib
        shadow_seg_med[unsafe_offset=base + local] = cv.med_idx
        shadow_valid[unsafe_offset=base + local] = Int8(1)

# ── VCM vertex merging: spatial hash grid over the LVC ───────────────────────
# Mirrors sppm.mojo's photon hash grid (_build_grid/_sppm_insert_photon/
# _sppm_reset_grid_cell) exactly, but keyed on BDPTVertex.pos instead of
# SPPMPhoton.pos, and using a SEPARATE parallel `merge_next` array for
# chaining rather than a field inside BDPTVertex itself (avoids touching
# BDPTVertex's layout/every other construction site in this file). Reuses
# sppm.mojo's _HSIZE bucket count and _hash_cell function directly -- no
# reason for the grid math itself to differ between the two use sites.

def _bdpt_reset_merge_cell(heads: Pointer[Int32, MutUntrackedOrigin], h: Int):
    heads[unsafe_offset=h] = Int32(-1)

def _bdpt_insert_merge_vertex[use_gpu: Bool](
    k: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
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
    if local_idx >= Int(lvc_path_len[unsafe_offset=lp_idx]):
        return
    var ix = Int(floor(lvc[unsafe_offset=k].pos.x * inv_cell))
    var iy = Int(floor(lvc[unsafe_offset=k].pos.y * inv_cell))
    var iz = Int(floor(lvc[unsafe_offset=k].pos.z * inv_cell))
    var h = _hash_cell(ix, iy, iz)
    comptime if use_gpu:
        var old = Atomic._xchg(heads.unsafe_offset(h), Int32(k))
        merge_next[unsafe_offset=k] = old
    else:
        merge_next[unsafe_offset=k] = heads[unsafe_offset=h]
        heads[unsafe_offset=h] = Int32(k)

def _bdpt_build_merge_grid(
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    n_light_paths: Int,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
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

comptime _VCM_RADIUS_FRACTION = Float32(0.03)   # initial radius as a fraction of the scene bounding sphere
comptime _VCM_RADIUS_ALPHA = Float32(2.0) / Float32(3.0)  # Georgiev 2012's typical choice

@always_inline
def vcm_merge_radius(scene_radius: Float32, si: Int) -> Float32:
    """The progressive VCM merge radius for sample `si` (0-based).

        r_i = 0.03 * scene_radius / (i+1)^(0.5*(1-alpha))

    Hachisuka & Jensen 2008 via Georgiev et al. 2012 Eq. 11: ONE global
    radius shared by every pixel this sample, shrinking monotonically --
    distinct from sppm.mojo's per-pixel Knaus-Zwicker scheme.

    Shared because this was written out THREE times (the CPU driver and
    two GPU drivers) with the fraction and the exponent hand-copied into
    each. That is not hypothetical drift: an experiment that changed only
    the CPU copy produced a perfect no-op and nearly sent the 2026-09-18
    merge-leak investigation down the wrong path, because the --gpu render
    under test was reading a different constant entirely."""
    return (scene_radius * _VCM_RADIUS_FRACTION
            / pow(Float32(si + 1), Float32(0.5) * (Float32(1) - _VCM_RADIUS_ALPHA)))


@always_inline
def _bdpt_merge_from_cache(
    cv: BDPTVertex,
    ref sd: SceneDescriptor2_C,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    r2: Float32,
    norm: Float32,
    mis_vc_weight_factor: Float32,
) -> SpectralSample:
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

    VCM Stage 2c/2d/153/hair: real per-candidate MIS weight (Georgiev et
    al. 2012 / SmallVCM's RangeQuery::Process, vertexcm.hxx:129-166,
    verified against the reference source) is applied when both cv and lv
    are `_bdpt_vertex_mis_scoped` (diffuse/conductor/coated_conductor/
    hair/measured) -- the same scope _connect uses for its own connection
    weight, for the same reason (see that function's docstring)."""
    if cv.is_delta != Int32(0) or cv.is_surface == Int32(0):
        return SpectralSample(Float32(0))
    # The merge GRID is what this needs, so test the grid -- not, as the call
    # sites used to, whether THIS pixel's own paired light path happened to
    # store a vertex. Those are different questions, and conflating them cost
    # merging most of its energy (see the call sites). A caller that never
    # intends to merge passes dangling sentinels here, the _is_real_ptr
    # convention from geometry.mojo, and the old `path_len > 0` guard was
    # shielding them by accident.
    if not (_is_real_ptr(heads) and _is_real_ptr(merge_next) and _is_real_ptr(lvc)):
        return SpectralSample(Float32(0))
    # A vertex kind with no real pdf has no real MIS weight either -- the
    # weight below falls back to 1, and an unweighted merge summed with an
    # unweighted connect estimates 2I, not I. So such a vertex must not merge
    # AT ALL; connect alone is already its complete, correct estimate. The
    # volume branch of _bdpt_camera_path_bounce has always said exactly this
    # and gated its own call; the other call sites relied on `path_len > 0`
    # to do it by accident, and hoisting that gate (correctly, merging does
    # not depend on the paired light path) exposed them. coateddiffuse is the
    # one that bites: its coat-walk vertex is stored with dVCM = dVC = dVM = 0
    # and pdf_fwd = 1 placeholders, so it is deliberately out of scope --
    # furnace-coateddiffuse.vcm read 1.64 against an analytic 1.0 with it
    # merging unweighted. Giving that vertex REAL carries is its own task
    # (see the elegance backlog's item 9); until then it does not merge.
    if not _bdpt_vertex_mis_scoped(cv):
        return SpectralSample(Float32(0))
    var total = SpectralSample(Float32(0))
    var cix = Int(floor(cv.pos.x * inv_cell))
    var ciy = Int(floor(cv.pos.y * inv_cell))
    var ciz = Int(floor(cv.pos.z * inv_cell))
    for ddx in range(-1, 2):
        for ddy in range(-1, 2):
            for ddz in range(-1, 2):
                var h = _hash_cell(cix + ddx, ciy + ddy, ciz + ddz)
                var k = Int(heads[unsafe_offset=h])
                while k != -1:
                    var lv = lvc[unsafe_offset=k]
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
                    # BSSRDF exits excluded: a light-side exit vertex has no
                    # incoming ray (it was reached by a hop), so there is no
                    # photon direction to evaluate the camera vertex against.
                    # ... and the LIGHT vertex must be MIS-scoped too, for the
                    # reason this function's own docstring already gives about
                    # the CAMERA vertex: a kind with no real pdf has no real
                    # MIS weight, the weight below falls back to 1, and an
                    # unweighted merge summed with a weighted connect
                    # estimates more than I. That gate was enforced on cv (an
                    # early return) but not on lv, so the asymmetry let an
                    # unscoped light vertex merge at FULL weight. Reachable in
                    # practice: lobe_scoped's list is lambertian/ggx/hair/
                    # measured/coated_walk, so a diffuse_transmit photon is
                    # unscoped -- and barcelona-pavilion's foliage is 5
                    # diffusetransmission materials, sitting exactly over the
                    # shadowed regions that measured 2-3x too bright.
                    if lv.is_delta == Int32(0) and lv.is_surface == Int32(1) and lv.is_light == Int32(0) and lv.mat_kind != LobeKind.bssrdf and _bdpt_vertex_mis_scoped(lv):
                        var e = lv.pos - cv.pos
                        var dist2 = e.length_sq()
                        # Surface-compatibility guard: a distance-only gather
                        # counts a photon lying on a DIFFERENT surface (the
                        # adjacent wall, the far side of a thin panel) as if
                        # it were on this one. The leak grows with the gather
                        # radius, which here is 3% of the scene bounding
                        # sphere -- large in a room-sized scene.
                        var _ncmp = dot(cv.normal.to_simd(), lv.normal.to_simd())
                        # Gather in the tangent DISK, not the ball. The normal
                        # test above rejects a photon on a differently-oriented
                        # surface; it cannot reject one on a PARALLEL surface
                        # inside the radius -- a desk top over a shelf, a sill
                        # over a floor. Two lit parallel surfaces in one ball
                        # sum both their photons and normalise by ONE disk,
                        # pi r^2: up to 2x, radius-dependent, and impossible on
                        # a single flat quad, which is why the white furnace
                        # stayed exact while classroom read 2.5x pbrt with
                        # merging on and 0.985x with it off. A photon on THIS
                        # surface sits on its tangent plane to float precision;
                        # a tenth of the radius is generous.
                        # Disk-not-ball: the SHARED test, sppm.mojo's
                        # gather_disk_contains -- SPPM's gather now uses it too.
                        if _ncmp > Float32(0.7) and gather_disk_contains(e.to_simd(), dist2, r2, cv.normal.to_simd()):
                            var le_cv = _lobe_eval[want_pdfs=False](cv, lv.wo.to_simd(), sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, cv.wavelengths)
                            var f_cv = le_cv.f_cos
                            # MERGING TAKES THE BARE BSDF, NOT f*cos.
                            # _eval_vertex_spectral returns f*|cos| because
                            # that is what a CONNECTION needs -- the cosine
                            # belongs to its geometry term. A photon-density
                            # estimate must not apply it: the incident cosine
                            # is already carried by the photon AREAL DENSITY,
                            # grazing arrivals being proportionally rarer per
                            # unit area. Applying it again integrates cos^2
                            # where the reflection integral wants cos -- over
                            # a hemisphere exactly (2pi/3)/pi = 2/3, and
                            # merging alone measured 0.685 of the white
                            # furnace's analytic answer. SmallVCM splits it
                            # the same way: RangeQuery::Process uses the bare
                            # bsdfFactor, connections multiply by cosThetaGen.
                            # ... and THIS lobe's own cosine, which is not
                            # always |cos(dir, n)|: hair carries the FIBRE
                            # cosine and a volume carries none. Guessing it
                            # here -- as this did before _lobe_eval existed --
                            # divides hair by the wrong quantity entirely.
                            if le_cv.cos_used > Float32(1e-6):
                                f_cv = f_cv * (Float32(1.0) / le_cv.cos_used)
                            else:
                                f_cv = SpectralSample(Float32(0.0))
                            var w = Float32(1)
                            if _bdpt_vertex_mis_scoped(cv) and _bdpt_vertex_mis_scoped(lv):
                                var (camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w) = _bdpt_vertex_pdfs(cv, lv.wo.to_simd(), sd)
                                var w_light = lv.dVCM * mis_vc_weight_factor + lv.dVM * camera_bsdf_dir_pdf_w
                                var w_camera = cv.dVCM * mis_vc_weight_factor + cv.dVM * camera_bsdf_rev_pdf_w
                                w = Float32(1) / (w_light + Float32(1) + w_camera)
                            total += f_cv * lv.beta * w
                    k = Int(merge_next[unsafe_offset=k])
    return total * cv.beta * norm

# ── Trace one camera subpath, connecting to the shared cache inline ─────────


def _bdpt_trace_camera_and_connect[use_gpu: Bool](
    r2c:     Pointer[Float32, MutUntrackedOrigin],
    c2w:     Pointer[Float32, MutUntrackedOrigin],
    px:      Int, py:      Int,
    ref sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    lvc:     Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:  Int,
    path_len: Int,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    px_scale: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_f: Float32,
    pass_wl: SampledWavelengths,
    film_filter: FilmFilter,
    start_med_idx: Int32 = Int32(-1),
) -> Tuple[SpectralSample, RGB]:
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

    # SHARED WITH THE WAVEFRONT GPU DRIVER -- see _bdpt_trace_light_path's
    # matching comment. This body was a byte-for-byte copy of
    # _bdpt_camera_path_init + _bdpt_camera_path_bounce (988 lines against
    # 978, 599 identical non-comment lines). Both designs now run the same
    # step: this one loops over it inline, the wavefront driver launches it
    # once per depth level with the loop-carried state -- including the
    # `total`/`first_alb` accumulators -- parked in a VCMCameraPathState_C.
    var st = _bdpt_camera_path_init[use_gpu](
        r2c, c2w, px, py, pcg, px_scale, n_light_paths_f, pass_wl, film_filter, start_med_idx)
    var ro = st.ro
    var rd = st.rd
    var beta = st.beta
    var total = st.total
    var first_alb = st.first_alb
    var n_verts = Int(st.n_verts)
    var n_bounces = Int(st.n_bounces)
    var cur_med_idx = st.cur_med_idx
    var dvcm_carry = st.dvcm
    var dvc_carry = st.dvc
    var dvm_carry = st.dvm
    var last_bsdf_pdf = st.last_bsdf_pdf
    var mis_null_dist = st.mis_null_dist
    var current_dielectric_ior = st.current_dielectric_ior
    var previous_dielectric_ior = st.previous_dielectric_ior
    var wavelengths = SampledWavelengths(st.wl0, st.wl1, st.wl2, st.wl3, st.wl_pdf)
    if st.active == Int8(0):
        return (total, first_alb)

    for _ in range(_BDPT_MAX_DEPTH):
        # The same intersect step _bdpt_camera_path_intersect_gpu performs;
        # kept at the call site because it is the Vulkan RT swap point.
        var ray = Ray_C(ro, rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        # A miss (including its infinite-light escape credit) is handled
        # inside the step, so the old top-of-loop miss block is gone.
        if not _bdpt_camera_path_bounce[use_gpu](
            sd, pcg, has_med, scratch[unsafe_offset=0], scratch, lvc, lp_idx, path_len,
            merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
            mis_vc_weight_factor, mis_vm_weight_factor,
            ro, rd, beta, total, first_alb, n_verts, n_bounces, cur_med_idx,
            dvcm_carry, dvc_carry, dvm_carry, last_bsdf_pdf, mis_null_dist,
            current_dielectric_ior, previous_dielectric_ior, wavelengths,
            Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14]), px_scale):
            break

    return (total, first_alb)

@fieldwise_init
struct VCMCameraPathState_C(TrivialRegisterPassable):
    """Task #163 stage 4: persistent per-camera-path state carried across
    separate wavefront-staged GPU kernel launches (`_bdpt_camera_path_init`
    then one `_bdpt_camera_path_bounce` call per bounce), the camera-path
    counterpart to `VCMLightPathState_C` (see that struct's docstring for
    the general convention). Unlike the light-path side, `total`/`first_alb`
    are running ACCUMULATORS carried across every bounce, not just
    per-bounce scratch -- the host loop reads them once `active` drops to 0,
    the same value the original single-function loop would have returned."""
    var ro: Point3f
    var rd: Vec3f
    var beta: SpectralSample
    var total: SpectralSample
    var first_alb: RGB
    var dvcm: Float32
    var dvc: Float32
    var dvm: Float32
    var n_verts: Int32
    var n_bounces: Int32
    var cur_med_idx: Int32
    var last_bsdf_pdf: Float32
    var active: Int8
    var pcg_state: UInt64
    var pcg_inc: UInt64
    var wl0: Float32
    var wl1: Float32
    var wl2: Float32
    var wl3: Float32
    var wl_pdf: Float32
    # Distance travelled through null interfaces since the last real
    # scattering event -- same role as PathState_C.mis_null_dist. The
    # interface branch resets `ro` to the boundary it crossed, so a later
    # emitter hit's t_hit measures from the boundary, not from the vertex
    # whose sample generated the direction; the MIS pdf needs the latter.
    var mis_null_dist: Float32
    # Touching-dielectric IOR depth-2 stack for _dielectric_bounce (see that
    # function's docstring, sppm.mojo) -- same role and convention as
    # PathState_C.current_dielectric_ior/previous_dielectric_ior
    # (geometry.mojo). Both start at vacuum (1.0).
    var current_dielectric_ior: Float32
    var previous_dielectric_ior: Float32

def _bdpt_camera_path_init[use_gpu: Bool](
    r2c:     Pointer[Float32, MutUntrackedOrigin],
    c2w:     Pointer[Float32, MutUntrackedOrigin],
    px:      Int, py:      Int,
    mut pcg: PCG32,
    px_scale: Float32,
    n_light_paths_f: Float32,
    pass_wl: SampledWavelengths,
    film_filter: FilmFilter,
    start_med_idx: Int32 = Int32(-1),
) -> VCMCameraPathState_C:
    """Task #163 stage 4: camera-ray generation + MIS-origin setup half of
    `_bdpt_trace_camera_and_connect` (bdpt.mojo:830-886), split out to seed
    a `VCMCameraPathState_C` for the wavefront-staged bounce loop instead of
    falling straight into an inline `for` loop. Byte-for-byte copy of that
    function's pre-loop body -- see its own docstring/VCM Stage 2b comments
    for the cameraPdfW derivation, not repeated here. Unlike
    `_bdpt_light_path_init`, there is no early-inactive case (a camera
    subpath always starts active, even in the degenerate zero-length-ray
    edge case the original code silently tolerates) -- `active` is always 1."""
    # JITTER. This was `px + 0.5` -- the pixel CENTRE, identically for every
    # sample. VCM was the only integrator that did not jitter: the path tracer
    # uses `px + 0.5 + deltaX` through the scene's reconstruction filter
    # (sampling.mojo, gen_primary_ray_state) and SPPM uses
    # `px + pcg.next_float()` (sppm.mojo). So VCM had no anti-aliasing at all
    # and all of its spp were perfectly correlated in the film dimension.
    #
    # It also turned any first-hit-determined black into a PERMANENT black
    # pixel, since all 64 samples traced the identical ray. On
    # barcelona-pavilion that was 1587 pixels (1.98% of the lit image) stuck
    # at exactly 0.0, identical under --seed 1 and --seed 7 -- foliage, where
    # the centre ray lands on a leaf whose alpha cut-out we do not parse (see
    # `Shape "texture alpha"`, still unimplemented) so the quad is opaque
    # black. Rendering at 2x and downsampling drops that count to ZERO, which
    # is what identified the sampling as the amplifier rather than a
    # geometric hole or a NaN.
    #
    # Uniform jitter, matching SPPM. NOT the path tracer's Gaussian
    # reconstruction filter: sharing that needs the filter parameters plumbed
    # into this kernel, and SPPM already boxes, so this leaves VCM consistent
    # with one of the two rather than inventing a third behaviour. Unifying
    # all three on the real filter is the follow-up.
    #
    # FOLLOW-UP DONE: all three integrators now draw the sub-pixel position
    # through the scene's PixelFilter with the same sampler (sampling.mojo's
    # filter_sample_2d, pbrt-v4's kernel shapes). The t=1 splats are spread
    # over the same footprint, see _bdpt_splat_filtered.
    var u_fx = pcg.next_float()
    var u_fy = pcg.next_float()
    var (dfx, dfy) = film_filter_offset(u_fx, u_fy, film_filter)
    var fX = Float32(px) + Float32(0.5) + dfx
    var fY = Float32(py) + Float32(0.5) + dfy
    var (rd, ro, _cl) = camera_ray_from_film_xy(fX, fY, r2c, c2w)

    # VCM Stage 2b: real per-vertex MIS state for the eye subpath (see
    # project_vcm_stage2_mis_derivation memory). cameraPdfW derived by
    # analogy with SmallVCM's pinhole-camera imageToSolidAngleFactor,
    # using px_scale (world-space size of one pixel at unit distance along
    # the camera forward axis — same quantity the plain path tracer's mip
    # LOD already uses, pipeline.mojo) as the "pixel area at distance 1"
    # convention: cameraPdfW = 1/(px_scale² × cosθ³), cosθ = angle between
    # this ray and the camera forward axis. c2w's rotation preserves dot
    # products, so dotting the WORLD-space direction with the camera's own
    # forward axis in world space (c2w's z-column) equals what the raw
    # camera-space z-component would have been before that rotation -- no
    # need to keep the pre-rotation cx/cy/cz around just for this.
    var cos_theta_at_camera = abs(
        rd.x * c2w[unsafe_offset=8] + rd.y * c2w[unsafe_offset=9] + rd.z * c2w[unsafe_offset=10])
    var camera_pdf_w = Float32(1) / max(
        px_scale * px_scale * cos_theta_at_camera * cos_theta_at_camera * cos_theta_at_camera,
        Float32(1e-12))
    var dvcm_carry = n_light_paths_f / camera_pdf_w
    var dvc_carry = Float32(0)
    var dvm_carry = Float32(0)

    var n_verts = 0
    var n_bounces = 0  # total surface hits including glass (for _dielectric_bounce entering logic)
    var beta = SpectralSample(Float32(1))
    var cur_med_idx = start_med_idx
    var total = SpectralSample(Float32(0))
    var first_alb = RGB(Float32(0))  # denoiser albedo AOV -- set at the first stored vertex, below
    # ONE hero-wavelength set per spp PASS, shared by every camera AND light
    # subpath in it, rather than one per subpath. This is forced by the flip
    # to spectral transport: a connection multiplies a camera vertex's beta
    # by a light vertex's flux, and a merge does the same across ARBITRARY
    # light paths -- lane i of one only means the same thing as lane i of the
    # other if both were traced at identical wavelengths. Per-subpath
    # sampling made that false for every connection and every merge. Sharing
    # per pass correlates wavelengths across pixels within one sample, which
    # costs nothing in expectation (the pass index still decorrelates across
    # spp) and is what photon-mapping-family renderers do for exactly this
    # reason.
    var wavelengths = pass_wl
    # pdf (solid angle) of the cosine-weighted diffuse bounce that produced
    # the CURRENT `rd`, used to MIS-weight this ray's eventual infinite-light
    # miss-escape contribution against the NEE-to-infinite-light sample taken
    # at the vertex that generated it (see the diffuse branch below and the
    # miss handler). -1 = no competing NEE strategy exists for whatever
    # generated this ray (primary ray, or a conductor/dielectric/volume
    # bounce — none of which do infinite-light NEE in this function) → full
    # weight, no MIS needed. Mirrors shading.mojo's lastBsdfPdf bookkeeping.
    var last_bsdf_pdf = Float32(-1)


    return VCMCameraPathState_C(
        ro, rd, beta, total, first_alb, dvcm_carry, dvc_carry, dvm_carry,
        Int32(n_verts), Int32(n_bounces), cur_med_idx, last_bsdf_pdf, Int8(1),
        pcg.state, pcg.inc,
        wavelengths.lambda0, wavelengths.lambda1, wavelengths.lambda2, wavelengths.lambda3, wavelengths.pdf,
        Float32(0.0),
        Float32(1.0), Float32(1.0),   # current_dielectric_ior, previous_dielectric_ior (vacuum)
    )

def _bdpt_camera_path_bounce[use_gpu: Bool](
    ref sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    inter: Intersection_C,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    lvc:     Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:  Int,
    path_len: Int,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    mut ro: Point3f,
    mut rd: Vec3f,
    mut beta: SpectralSample,
    mut total: SpectralSample,
    mut first_alb: RGB,
    mut n_verts: Int,
    mut n_bounces: Int,
    mut cur_med_idx: Int32,
    mut dvcm_carry: Float32,
    mut dvc_carry: Float32,
    mut dvm_carry: Float32,
    mut last_bsdf_pdf: Float32,
    mut mis_null_dist: Float32,
    mut current_dielectric_ior: Float32,
    mut previous_dielectric_ior: Float32,
    wavelengths: SampledWavelengths,
    # Camera position + pixel angular size, for the bump/normal-map footprint
    # at each vertex (shading.mojo's _camera_approx_footprint). VCM tracks no
    # ray cone of its own, unlike the path tracer's PathState_C.cone_len, so
    # this is pbrt's Approximate_dp_dxy convention: footprint = pixel size x
    # distance from the camera. Using it on BOTH subpaths is what makes a
    # merge pair agree about the surface it is standing on.
    cam_pos: Vec3f,
    px_scale: Float32,
    # Task #163 stage 5: when set, the DIFFUSE branch's connect step queues
    # its shadow rays into these buffers (one _BDPT_MAX_VERTS-sized slice
    # per pixel, indexed like the LVC itself) instead of resolving them
    # inline via software BVH -- see _bdpt_connect_to_cache_deferred's own
    # docstring. Every OTHER material branch's connect (and every NEE/MNEE
    # shadow ray anywhere) is UNCHANGED regardless of this flag -- a
    # deliberately narrow first slice, not "all shadow rays on Vulkan RT".
    defer_shadow_rays: Bool = False,
    shadow_rays: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin] = Pointer[SpectralSample, MutUntrackedOrigin].unsafe_dangling(),
    shadow_valid: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
) -> Bool:
    """Task #163 stage 4: wavefront-staged variant of ONE bounce iteration of
    `_bdpt_trace_camera_and_connect`'s main loop (bdpt.mojo:887-1684), the
    camera-path counterpart to `_bdpt_light_path_bounce` (see that function's
    docstring for the general rationale -- same design, applied to the
    larger of the two subpath loops). `scratch` is STILL a live parameter
    here, unlike the light-path bounce function: shadow rays (NEE/connect/
    merge/MNEE, the majority of ray casts per sample) stay on the existing
    software-BVH `any_hit`-style tests inside `_bdpt_nee_contribute`/
    `_bdpt_connect_to_cache`/`_bdpt_merge_from_cache`/
    `_bdpt_mnee_diffuse_area_light`, unchanged -- only this function's own
    PRIMARY/bounce ray intersect is externalized to the caller, per the
    user-confirmed stage 4 scope (AskUserQuestion, 2026-07-12: "shadow rays
    stay on software BVH in this first cut").

    The body below is a byte-for-byte copy of the original loop body
    (mechanically extracted, not retyped), with the same class of
    control-flow-shape changes `_bdpt_light_path_bounce` used: every
    outer-loop `break` (9 sites: direct hit on an emissive analytic sphere,
    direct hit on an area-light triangle/curve, coat total-internal-
    reflection-like case, coat recycling walk absorbed, conductor invalid
    sample, 3x measured-BxDF failure, the unhandled-material-type
    catch-all) becomes `return False`; the loop's 2 outer `continue` sites
    (volume free-flight scatter; rough-coat reflect) become `return True`;
    the coat recycling walk's OWN inner `for depth in
    range(MAX_COAT_DEPTH):` loop keeps its own real `break` statements
    (RR kill, exit-direction found) untouched, since those exit the INNER
    walk loop, not this function. The original miss-handling block (evaluate
    infinite-light MIS against `last_bsdf_pdf`, add to `total`) is preserved
    verbatim as this function's own top-of-body miss check, since the
    caller now supplies `inter` pre-computed instead of this function
    calling `traverse_bvh2_core`/`test_spheres` itself.

    `total`/`first_alb`/`beta`/`n_verts`/`n_bounces`/`cur_med_idx`/
    `dvcm_carry`/`dvc_carry`/`dvm_carry`/`last_bsdf_pdf` are `mut`
    PARAMETERS instead of function-local variables persisted implicitly
    across loop iterations -- the caller is a persistent per-camera-path
    state struct that survives across separate kernel launches, one call to
    this function per bounce. `total`/`first_alb` in particular ACCUMULATE
    across calls (unlike the light-path side, which has no equivalent
    running accumulator) -- the host loop reads them from the state after
    the path goes inactive, exactly like it would read a local variable
    after the original single-function loop returned.

    Returns True if the camera path should continue to another bounce,
    False if it has terminated (escaped the scene, hit an emitter, exhausted
    a material's valid-sample conditions, or hit an unhandled material
    type)."""
        if n_verts >= _BDPT_MAX_VERTS:
            return False   # mirrors the original loop's top-of-iteration guard
        if inter.hit == Int8(0):
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ilight = sd.infiniteLights[unsafe_offset=inf_i]
                var (Le, pdf_light_here) = _eval_infinite_light_and_pdf(ilight, rd)
                var mis_w = Float32(1)
                if last_bsdf_pdf == PDF_DROP_DIRECT:
                    # A scatter whose direct term NEE already reported (the
                    # rough-coat exit ray): indirect only. Same contract PT's
                    # handlers keep; bdpt's gate below let this through as a
                    # zero pdf at near-full weight, double-counting against
                    # the coat's full-weight per-iteration NEE.
                    mis_w = Float32(0)
                elif n_verts > 0 and pdf_light_here > Float32(0):
                    # NOT `last_bsdf_pdf >= 0`, which is what this used to be.
                    # That test excluded every path whose last event was DELTA
                    # (the dielectric branch sets last_bsdf_pdf = -1), handing
                    # the escape a hard-coded weight of 1 -- while the diffuse
                    # vertex further back had already reported the same
                    # transport through connect/merge. Both counted in full:
                    # Scenes/furnace/dielectric-inert.pbrt read 1.1708 at the
                    # default light-path count and converged to 1.9665, against
                    # an analytic 1.0.
                    #
                    # No specular special case is needed, and SmallVCM does not
                    # have one either (GetLightRadiance weights unconditionally
                    # past the first hit): a delta bounce sets dvcm_carry = 0,
                    # and a zero dVCM already removes the NEE term from the
                    # weight below on its own, while dVC keeps discounting
                    # connect and merge -- which is the part the old gate threw
                    # away. `n_verts > 0` now carries the one case that IS
                    # unweighted: NO REAL VERTEX HAS BEEN STORED YET, so no
                    # other strategy can have reported this transport -- a
                    # primary ray straight into the environment, or one that
                    # only ever hit DELTA surfaces (mirror, glass), neither of
                    # which stores a vertex. That is a weight-1 case, not a
                    # "no pdf" case, and conflating the two is what hid the bug.
                    #
                    # It must be n_verts and NOT n_bounces: the diffuse branch
                    # stores a vertex without incrementing n_bounces, so an
                    # n_bounces test reads an ordinary diffuse path as a
                    # primary ray and hands every furnace escape weight 1 --
                    # measured, every VCM env-lit furnace cell went ~1.01 to
                    # ~1.45 before this was corrected.
                    # Balance heuristic over EVERY strategy, matching the rest
                    # of this file -- vcm_env_escape_weight, derived exactly in
                    # Scenes/vcm_env_mis_derivation.py. The power heuristic it
                    # replaces gave the escape 0.80 where balance over all
                    # strategies gives 0.54, leaving no share for merging or
                    # t=1. dvcm/dvc_carry are the POST-SCATTER carries from the
                    # last real vertex (this handler runs before the per-hit
                    # arrival update), which is exactly what that weight wants.
                    var (_c_esc, r_esc) = _scene_bounding_sphere(sd)
                    var emis_esc = pdf_light_here / max(_bdpt_n_lights(sd) * PI * r_esc * r_esc, Float32(1e-12))
                    mis_w = vcm_env_escape_weight(pdf_light_here, emis_esc,
                                                  dvcm_carry, dvc_carry)
                total += beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (Le).r, (Le).g, (Le).b, wavelengths) * mis_w
            return False   # nothing hit -- path escapes the scene
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
            var med = sd.mediums[unsafe_offset=Int(cur_med_idx)]
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            # density field. This call site used to be the homogeneous one
            # unconditionally, which rendered every "uniformgrid"/"nanovdb"/
            # "cloud" medium as uniform density-1 fog -- bunny-cloud came out a
            # featureless sphere with no bunny in it. See sample_free_flight.
            var ff = sample_free_flight(
                med, sd.grids, sd.nvdbGrids, Vec3f(ro.x, ro.y, ro.z), rd, t_hit, pcg)
            if ff.collided:
                # Chromatic collision weight -- the ratio to the sampled (red)
                # channel. Without it a chromatic medium is biased at every
                # scattering event; exactly 1 for a grey medium. Applied to
                # `beta` BEFORE the vertex stores it, so the stored throughput
                # is the one arriving at the vertex.
                beta *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # Volume scatter vertex
                var sp = ro + rd*ff.t_free
                var v = _null_vertex()
                v.pos = sp
                # `beta`, NOT beta*albedo -- a vertex's beta is the throughput
                # ARRIVING at it; connect time applies the vertex's own
                # response separately (v.alb/(4*pi)). Baking albedo in here
                # too double-counted it (see docs/09_volumetric_media.md,
                # "VCM/BDPT volume connections": 0.785x -> 0.983x fix).
                v.beta = beta
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.pdf   # the density actually sampled from; under hero-wavelength MIS this is a lane MIXTURE, not sig_t's lone exponential
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                if n_verts == 0: first_alb = ff.albedo
                # NOT `n_verts += 1`. _BDPT_MAX_VERTS is the light-vertex
                # CACHE's per-path slot count, and the camera subpath stores
                # nothing in that cache (only _bdpt_light_path_bounce calls
                # _bdpt_store_lvc_vertex). Counting volume scatters against it
                # capped a camera walk at 10 scatter events -- far too few for
                # a dense medium, where a diffusive walk needs tens of orders
                # to converge (the path tracer needs maxdepth ~33 on the same
                # scene). Volume scatters are bounded by _BDPT_MAX_DEPTH loop
                # iterations instead, which is what this branch's own "no
                # vertex stored this bounce" return comment already implied.
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
                # Volume-scatter NEE: distant/point/sphere/infinite lights.
                # _bdpt_connect_to_cache above only reaches AREA lights -- the
                # only kind _bdpt_light_path_init seeds the light-vertex cache
                # from (its own `n_lights` count is area+distant+infinite+point,
                # but only area lights get a real lv0 origin vertex placed in
                # the cache; see its docstring). Every other light type
                # therefore illuminated a scatter vertex only through the
                # tiny-probability event of an isotropic-scattered ray
                # randomly re-hitting it (the sphere case of which was ALSO
                # unweighted until the fix just above this block) -- both far
                # too dark and far too noisy, the same failure this file's own
                # docs/09_volumetric_media.md chapter describes for the
                # wavefront integrator's pre-fix state. Mirrors gpu.mojo's
                # _volume_nee_light (that integrator's equivalent): phase
                # (isotropic, INV_FOUR_PI -- this file's volume scatter is
                # hardcoded isotropic, see the uniform-sphere sample just
                # below, so g plays no role here yet, a separate pre-existing
                # gap) x albedo x Li x reciprocal power-heuristic MIS / pdf.
                # See project_sphere_light_nee_bug memory, "STILL OPEN: sphere
                # lights + media under --vcm read 0.66x".
                # Albedo is a per-channel COEFFICIENT (sigma_s/sigma_t), not a
                # reflectance -- band-pick it. spec_refl was used here, and
                # RGB(a,a,a) does not upsample to a in every lane, so even a
                # grey medium picked up a D65-shaped tint (see
                # docs/02_spectra_and_color.md, "A coefficient is not a color").
                var alb_spec_v = rgb_bands_to_spectral_sample((ff.albedo).r, (ff.albedo).g, (ff.albedo).b, wavelengths)
                var phase_alb_v = alb_spec_v * INV_FOUR_PI
                for li_v in range(_bdpt_simple_light_count(sd)):
                    var ls_v = _bdpt_sample_simple_light(sd, li_v, sp.to_simd(), pcg)
                    if ls_v.valid and ls_v.pdf > Float32(0):
                        var li_spec_v = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ls_v.Li).r, (ls_v.Li).g, (ls_v.Li).b, wavelengths)
                        var mis_v = Float32(1) if ls_v.is_delta else power_heuristic(ls_v.pdf, INV_FOUR_PI)
                        var w_v = phase_alb_v * li_spec_v * (mis_v / ls_v.pdf)
                        total += _bdpt_nee_contribute(beta, w_v, ls_v, sp, Vec3f(Float32(0)), cur_med_idx, sd, scratch, wavelengths, Float32(0))
                for inf_v in range(Int(sd.infiniteLightCount)):
                    var ls_infv = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_v], Point2f(pcg.next_float(), pcg.next_float()))
                    if ls_infv.valid and ls_infv.pdf > Float32(0):
                        var li_spec_infv = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ls_infv.Li).r, (ls_infv.Li).g, (ls_infv.Li).b, wavelengths)
                        var mis_infv = Float32(1) if ls_infv.is_delta else power_heuristic(ls_infv.pdf, INV_FOUR_PI)
                        var w_infv = phase_alb_v * li_spec_infv * (mis_infv / ls_infv.pdf)
                        total += _bdpt_nee_contribute(beta, w_infv, ls_infv, sp, Vec3f(Float32(0)), cur_med_idx, sd, scratch, wavelengths, Float32(0))
                # Continuation beta = prev × alb_s (same as stored vertex beta)
                beta *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ff.albedo).r, (ff.albedo).g, (ff.albedo).b, wavelengths)
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                last_bsdf_pdf = _VOL_PHASE_HIT   # see the sentinel's definition
                mis_null_dist = Float32(0)     # this vertex is the new origin
                return True   # volume free-flight scatter: no vertex stored this bounce, path continues
            else:
                # Pass-through weight (a per-channel RATIO -- see
                # sample_homogeneous_free_flight; the sampled channel's
                # transmittance is already carried by the pass-through
                # probability, so multiplying the FULL Beer-Lambert factor
                # here double-counted it).
                beta *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # MIS: dVCM must equal 1/P(prev -> this vertex), and reaching a
                # SURFACE through a medium carries a survival factor
                # FF = exp(-sigma_t * d) that the vacuum recursion above
                # (`dvcm_carry *= t_hit*t_hit`) does not include. So dVCM takes
                # 1/FF. dVC needs nothing here: its second-order term carries
                # FF_a/FF_b, and for a surface->surface edge both endpoints
                # contribute the same survival factor, so the ratio is exactly 1.
                # Verified against a brute-force enumeration of every strategy's
                # full path pdf -- see Scenes/vcm_surface_ff_gap.py, where the
                # uncorrected recursion's worst relative error runs 8e-16 in
                # vacuum but 0.27 at sigma_t=0.1, 12 at 1.0 and 1e5 at 4.0.
                # Identically 1 when sigma_t = 0, so this is inert outside media.
                var ff_exp = -log(max(ff.pdf, Float32(1e-30)))   # optical depth of the density actually sampled from (see FreeFlight.pdf)
                if ff_exp > Float32(60.0): ff_exp = Float32(60.0)  # defensive: e^-60 pass-through never occurs
                dvcm_carry *= exp(ff_exp)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
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
            var sph_hit = sd.spheres[unsafe_offset=Int(inter.primId.id1)]
            if sph_hit.isAreaLight != Int8(0):
                var mis_w_sph_hit = Float32(1)
                if last_bsdf_pdf == PDF_DROP_DIRECT:
                    # Direct term already reported by NEE: indirect only, so
                    # this hit contributes nothing. Was `pass`, which left the
                    # weight at 1 -- see the area-light handler below.
                    mis_w_sph_hit = Float32(0)
                elif last_bsdf_pdf >= Float32(0):
                    # The competing NEE strategy was taken at the vertex that
                    # GENERATED this ray, so its cone pdf must be measured from
                    # there -- `ro` -- exactly as the area-light case below
                    # measures its own pdf with `t_hit` from the same origin.
                    # This used to measure from `hit`, the point ON the sphere,
                    # where |center - hit| IS the radius by construction, so
                    # sin2_max was identically 1, the `< 1` guard below always
                    # failed and mis_w stayed 1: the emitter hit took FULL
                    # weight against an already-full-weight NEE and every
                    # sphere light was counted TWICE. Measured on a sphere
                    # light over a diffuse floor: --vcm read 1.96x pbrt, and
                    # the emitter hit alone accounted for 0.121 of the correct
                    # 0.130 where MIS should have given it ~3.6e-5.
                    var to_c_hit = sph_hit.center - ro
                    var dc_sq_hit = to_c_hit.length_sq()
                    var sin2_max_hit = sph_hit.radius * sph_hit.radius / dc_sq_hit
                    if sin2_max_hit < Float32(1):
                        var cos_max_hit = sqrt(Float32(1) - sin2_max_hit)
                        var solid_angle_hit = Float32(2) * PI * (Float32(1) - cos_max_hit)
                        # No 1/count factor -- see _sample_sphere_light_nee,
                        # whose pdf this must match exactly for MIS to be right.
                        var pdf_light_hit = Float32(1) / solid_angle_hit
                        mis_w_sph_hit = power_heuristic(last_bsdf_pdf, pdf_light_hit)
                elif last_bsdf_pdf == _VOL_PHASE_HIT:
                    # Same missing-case bug as the area-light branch above
                    # used to have: an isotropic-phase-sampled ray landing
                    # directly on a sphere light took FULL weight here with
                    # no competing-strategy reduction at all (this elif
                    # simply didn't exist). Same pdf convention as the
                    # `>= 0` branch, just weighed against the phase pdf
                    # (INV_FOUR_PI) instead of a surface BSDF's.
                    var to_c_vhit = sph_hit.center - ro
                    var dc_sq_vhit = to_c_vhit.length_sq()
                    var sin2_max_vhit = sph_hit.radius * sph_hit.radius / dc_sq_vhit
                    if sin2_max_vhit < Float32(1):
                        var cos_max_vhit = sqrt(Float32(1) - sin2_max_vhit)
                        var solid_angle_vhit = Float32(2) * PI * (Float32(1) - cos_max_vhit)
                        var pdf_light_vhit = Float32(1) / solid_angle_vhit
                        mis_w_sph_hit = power_heuristic(INV_FOUR_PI, pdf_light_vhit)
                total += beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (sph_hit.emission).r, (sph_hit.emission).g, (sph_hit.emission).b, wavelengths) * mis_w_sph_hit
                return False   # direct hit on emissive analytic sphere -- terminates the path

        # Mix material: stochastically resolve to one of two sub-materials
        # (mirrors shading.mojo's shade_mix) before any type dispatch below —
        # packing/guard-against-mix-of-mix convention identical to shade_mix.
        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[unsafe_offset=mix_chosen]
            mat_idx = mix_chosen  # keep in sync with the resolved sub-material (hair needs the real index to re-fetch at connect time)
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

        if mat.type == MatKind.area_light:
            # Direct hit on a triangle/curve area light — same MIS-against-
            # last_bsdf_pdf treatment as the sphere case above. id1 is the
            # AreaLight_C index directly for a type==3 (area-light-triangle)
            # hit, per pbrt_parser.mojo's own PrimId_C encoding.
            # Which light, what it emits, and which side of it is lit all come
            # from the SHARED resolvers (shading.mojo's area_light_hit_cos /
            # curve_light_hit), not from _geom_normal and areaLights[id1]. See
            # their header: the raw winding normal disagreed with the side the
            # light sampler emits from, and for a curve id1 is not an
            # AreaLight_C index at all.
            var al_emission: RGB
            var al_area = Float32(0)
            var cos_l_hit: Float32
            var is_curve = inter.primId.type == Int8(5)
            if is_curve:
                var (al_ci, cos_c) = curve_light_hit(inter, sd.curves, sd.areaLights,
                                                     Int(sd.areaLightCount), ray_dir)
                al_emission = mat.emission     # the curve's own emitter slot
                cos_l_hit = cos_c
                if al_ci >= 0:
                    al_area = sd.areaLights[unsafe_offset=al_ci].total_area
            else:
                var al_hit = sd.areaLights[unsafe_offset=Int(inter.primId.id1)]
                al_emission = al_hit.emission
                al_area = al_hit.total_area
                cos_l_hit = area_light_hit_cos(inter, sd.meshes, sd.instances, ray_dir)
            # A curve is a closed tube, so an unweighted (primary/delta) hit is
            # always on its outside -- credited regardless of the reconstructed
            # radial cosine, exactly as the path tracer does. A triangle light
            # is one-sided.
            if cos_l_hit > Float32(0) or (is_curve and last_bsdf_pdf < Float32(0) and last_bsdf_pdf != PDF_DROP_DIRECT):
                var mis_w_al_hit = Float32(1)
                if last_bsdf_pdf == PDF_DROP_DIRECT:
                    # NEE already reported this ray's direct term (the rough
                    # coat's exit ray) -- geometry.mojo's contract for this
                    # sentinel is "contribute indirect only", and an emitter
                    # hit IS the direct term. This used to `pass`, leaving the
                    # weight at 1: a full double count on every coat exit that
                    # landed on an emitter, which the env miss handler had
                    # already been fixed for.
                    mis_w_al_hit = Float32(0)
                elif last_bsdf_pdf >= Float32(0):
                    if cos_l_hit > Float32(0) and al_area > Float32(0):
                        var dist2_hit = t_hit * t_hit
                        var n_area_hit = Float32(max(Int(sd.areaLightCount), 1))
                        var pdf_light_al = dist2_hit / (cos_l_hit * n_area_hit * al_area)
                        mis_w_al_hit = power_heuristic(last_bsdf_pdf, pdf_light_al)
                    else:
                        mis_w_al_hit = Float32(0)
                elif last_bsdf_pdf == _VOL_PHASE_HIT:
                    # The competing strategy is the volume vertex's s=1
                    # connection to the light-source vertex, whose area pdf is
                    # 1/(total_area * n_lights) with n_lights counting EVERY
                    # light type (that is how _bdpt_light_path_init picks a
                    # light) -- spelled the same way here, since MIS is only
                    # right when both halves agree on the pdf. The distance is
                    # from the scattering vertex, not the current ray origin.
                    var d_vol = t_hit + mis_null_dist
                    var n_lights_hit = Float32(max(Int(sd.areaLightCount) + Int(sd.distantLightCount)
                                                   + Int(sd.infiniteLightCount) + Int(sd.pointLightCount), 1))
                    if cos_l_hit > Float32(0) and al_area > Float32(0):
                        var pdf_light_vol = d_vol * d_vol / (cos_l_hit * n_lights_hit * al_area)
                        mis_w_al_hit = power_heuristic(INV_FOUR_PI, pdf_light_vol)
                    else:
                        mis_w_al_hit = Float32(0)
                total += beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al_emission.r, al_emission.g, al_emission.b, wavelengths) * mis_w_al_hit
            return False   # direct hit on an area-light triangle/curve -- terminates the path

        # Every arm of the dispatch below except a null interface is a real
        # scattering event, and the null-interface distance accumulated on
        # the way here has served its purpose. A plain (non-chained) `if`,
        # deliberately: folding this into the elif chain below as its own
        # first arm silently swallowed every other branch (diffuse, conductor,
        # dielectric, interface's own logic, ...) -- an `if` that matches
        # takes over the ENTIRE following elif sequence, which is exactly
        # backwards from "reset unless interface". Caught by the smoketest
        # regressing (cpu-vcm 0.130 -> 0.106 on cornell-box, no medium at
        # all), which is why this comment is here: the failure mode is
        # completely silent otherwise -- it still compiles and still returns.
        if mat.type != MatKind.interface:
            mis_null_dist = Float32(0)

        if mat.type == MatKind.diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            # VCM Stage 2b: finish the per-bounce MIS correction (dist²
            # portion already applied above) -- see
            # _bdpt_trace_light_path's matching comment.
            var cos_fix = abs(dot(-ray_dir, gn))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix)
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
            # Bump/normal maps. bdpt.mojo applied NONE of them, on either
            # subpath, while the path tracer has since 2026-09 -- a textbook
            # instance of project_pt_only_feature_gaps, and the one that left
            # VCM's two halves standing on different geometry: the camera
            # vertex on a perturbed surface, the light vertex it merges with
            # on a flat one. Footprint from the camera-distance
            # approximation, the same one the light side uses, so a merge
            # pair filters the surface identically.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_geo = gn
            gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn, gn, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn_geo)
            v.shading_normal = vec3f(gn)
            v.beta = beta
            v.alb = eff_alb
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            # diffusetransmission has a transmit lobe; without its own
            # LobeKind the vertex is re-evaluated as opaque Lambertian and
            # loses exactly half its energy. mat_idx is required: the
            # transmittance lives in Material_C.emission, not on the vertex.
            if mat.type == MatKind.diffuse_transmit:
                v.mat_kind = LobeKind.diffuse_transmit
                v.mat_idx = Int32(mat_idx)
            v.pdf_fwd = Float32(1)  # unused by the uniform-subsample estimator
            v.wo = vec3f(-ray_dir)  # VCM Stage 2b: needed for _connect's reverse-pdf eval
            v.med_idx = cur_med_idx
            v.wavelengths = wavelengths
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            if n_verts == 0: first_alb = eff_alb
            n_verts += 1
            # Merging queries the GLOBAL photon grid and does not use this
            # pixel's own paired light path, so unlike the connect below it
            # must NOT be gated on that path having stored anything. It was,
            # and in a white furnace only ~32% of light paths hit the quad at
            # all, so ~68% of pixels skipped merging entirely: the estimator
            # delivered 0.109 against an analytic 0.5.
            total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
            if path_len > 0:
                # Task #163 stage 5: the diffuse branch's connect shadow
                # rays are the single highest-volume, cleanest shadow-ray
                # call site in VCM (always exactly one ray per stored
                # light-path vertex, present in every diffuse-heavy scene
                # regardless of light types used) -- when defer_shadow_rays
                # is set, queue them into the shadow-ray buffers instead of
                # resolving inline, for a later batched Vulkan RT dispatch.
                # Every other material branch's connect (and all NEE/MNEE
                # shadow rays) stays on the unchanged, inline,
                # software-BVH _bdpt_connect_to_cache path either way.
                if defer_shadow_rays:
                    _bdpt_connect_to_cache_deferred(v, sd, lvc, lp_idx, path_len, mis_vm_weight_factor, shadow_rays, shadow_pending, shadow_valid, shadow_seg_med)
                else:
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
            # distant/point/sphere via the shared sampler (pure loop collapse --
            # order was already distant,point,sphere, matching the iterator).
            for li_d in range(_bdpt_simple_light_count(sd)):
                var ls_i = _bdpt_sample_simple_light(sd, li_d, hit.to_simd(), pcg)
                # Distant lights get the VCM policy (emission = the disk's area
                # pdf over the pick, direct = 1 for a delta). Point and sphere
                # lights keep the default until their emission densities are
                # written down the same way -- noted, not silently assumed.
                var pol_i = mis_policy_power()
                if li_d < Int(sd.distantLightCount):
                    var (_c_i, r_i) = _scene_bounding_sphere(sd)
                    var le_i = _lobe_eval[want_pdfs=True](v, ls_i.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                    pol_i = MisPolicy(le_i.scoped, mis_vm_weight_factor, dvcm_carry, dvc_carry,
                                      Float32(1.0) / max(_bdpt_n_lights(sd) * PI * r_i * r_i, Float32(1e-12)),
                                      le_i.pdf_rev, False)
                var w_i = _nee_weight_simple_spectral(ls_i, v.mat_kind, eff_alb, Float32(0), gn, wo_d, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs), pol_i, v.mat_idx)
                total += _bdpt_nee_contribute(beta, w_i, ls_i, hit, gn, cur_med_idx, sd, scratch, wavelengths, Float32(0.0001), v.mat_kind == LobeKind.diffuse_transmit)
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ls_e = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float()))
                # The SAME shared helper every other material uses -- the
                # only difference is the MIS policy handed to it. Before
                # MisPolicy existed the weight was welded inside the helper,
                # so getting VCM's correct weight here meant hand-inlining
                # the whole throughput computation at this one site while the
                # other seven kept the path tracer's two-strategy heuristic.
                var (_c_e, r_e) = _scene_bounding_sphere(sd)
                var le_e = _lobe_eval[want_pdfs=True](v, ls_e.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                var pol_e = MisPolicy(True, mis_vm_weight_factor, dvcm_carry, dvc_carry,
                                      ls_e.pdf / max(_bdpt_n_lights(sd) * PI * r_e * r_e, Float32(1e-12)),
                                      le_e.pdf_rev, False)
                var w_e = _nee_weight_simple_spectral(ls_e, v.mat_kind, eff_alb, Float32(0), gn, wo_d, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs), pol_e, v.mat_idx)
                total += _bdpt_nee_contribute(beta, w_e, ls_e, hit, gn, cur_med_idx, sd, scratch, wavelengths, Float32(0.0001), v.mat_kind == LobeKind.diffuse_transmit)
            # Real MNEE for area lights behind glass (task #161) -- see
            # _bdpt_mnee_diffuse_area_light's docstring. Ordinary (non-glass)
            # area lights are deliberately left to connect/merge, unchanged.
            total += _bdpt_mnee_diffuse_area_light(sd, hit, gn, eff_alb, beta, pcg, wavelengths)
            # Sphere-shaped area lights (task #161 follow-up, 2026-07-13):
            # a completely separate list from sd.areaLights (see
            # _bdpt_mnee_sphere_light's docstring). Deliberately NOT a
            # `for` loop -- see that function's own investigation note for
            # the real GPU codegen bug (CUDA_ERROR_ILLEGAL_ADDRESS) a loop
            # here triggers on this task's own target scene. Manually
            # unrolled instead, capped at _MNEE_MAX_SPHERES (currently 4
            # -- update both together if that constant ever changes).
            var n_sph_mnee = Int(sd.sphereCount)
            if 0 < n_sph_mnee:
                total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 0, n_sph_mnee, wavelengths)
            if 1 < n_sph_mnee:
                total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 1, n_sph_mnee, wavelengths)
            if 2 < n_sph_mnee:
                total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 2, n_sph_mnee, wavelengths)
            if 3 < n_sph_mnee:
                total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 3, n_sph_mnee, wavelengths)

            # Cosine-weighted scatter direction
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            # diffusetransmission scatters to BOTH sides, with the lobe
            # picked by luminance -- the SAME split lobe_eval reports as the
            # density. Sampling one-sidedly while evaluating two-sidedly
            # leaves VCM's carries describing a path that was never built.
            # p_sel == 1 for every other material, so this is a no-op there
            # (and draws no extra random number).
            var p_sel = Float32(1.0)
            var bounce_n = gn
            var lobe_alb_dt = eff_alb
            if mat.type == MatKind.diffuse_transmit:
                var trans_dt = eff_alb if Int(mat.tex_idx) != -1 else mat.emission
                var pr_dt = eff_alb.luma()
                var pt_dt = trans_dt.luma()
                var tot_dt = max(pr_dt + pt_dt, Float32(1e-9))
                var take_refl = pcg.next_float() < pr_dt / tot_dt
                p_sel = max((pr_dt if take_refl else pt_dt) / tot_dt, Float32(1e-6))
                bounce_n = gn if take_refl else (gn * Float32(-1.0))
                lobe_alb_dt = eff_alb if take_refl else trans_dt
            rd = vec3f(_cosine_hemisphere_sample(bounce_n, u1, u2))
            ro = hit + rd*Float32(0.0002)
            last_bsdf_pdf = bxdf_pdf_diffuse(abs(dot(bounce_n, rd.to_simd()))) * p_sel
            # Update beta: f*cos/pdf = (alb/π)cos / (p_sel cos/π) = alb / p_sel.
            # The COLOUR stays a bounded reflectance and 1/p_sel rides along
            # as a plain scalar -- pushing alb/p_sel through spec_refl would
            # hit its [0,1] clamp (the same trap as _to_spec_weight).
            beta *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (lobe_alb_dt).r, (lobe_alb_dt).g, (lobe_alb_dt).b, wavelengths) * (Float32(1.0) / p_sel)
            # VCM Stage 2b: recursive continuation for the NEXT bounce --
            # see _bdpt_trace_light_path's matching diffuse-branch comment.
            var cos_theta_out = abs(dot(rd.to_simd(), bounce_n))
            # Both densities carry p_sel: the reverse of a transmission is a
            # transmission, so the same lobe probability applies each way --
            # which is exactly what lobe_eval's branch returns for fwd/rev.
            # cosThetaOut/bsdfDirPdfW is then PI/p_sel, not PI.
            var bsdf_rev_pdf_w = p_sel * cos_fix / PI
            var bsdf_dir_pdf_w = p_sel * cos_theta_out / PI
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                dvcm_carry, dvc_carry, dvm_carry, PI / p_sel, bsdf_dir_pdf_w, bsdf_rev_pdf_w,
                mis_vc_weight_factor, mis_vm_weight_factor)

        elif mat.type == MatKind.coated_diffuse:
            # VCM: coateddiffuse is a stochastic multi-bounce recycling walk
            # (see shading.mojo's shade_coated_diffuse, ported here almost
            # verbatim) with no closed-form joint pdf -- task #158, user
            # chose "full multi-tap recycling walk, left unweighted": ported
            # exactly so VCM's appearance matches the plain path tracer, but
            # every outcome (coat reflect, rough or smooth, and the base
            # exit) is stored/continued as mat_kind=4, which
            # _bdpt_vertex_mis_scoped deliberately excludes (same documented
            # gap as dielectric/volume) -- connect/merge still reach these
            # vertices via _eval_vertex's generic Lambertian fallback
            # (weight=1, no real MIS), never a full "no connection at all"
            # exclusion.
            var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var cos_fix = abs(dot(-ray_dir, gn))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix)
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            # Bump/normal maps -- see the diffuse branch above.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_geo = gn
            gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn, gn, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))

            var ior = mat.emission.r
            var coat_alpha = max(mat.roughU, mat.roughV)
            var is_rough_coat = coat_alpha > Float32(0.001)
            var wo = -ray_dir
            # THE shared layered-BSDF walk (bxdf.mojo) -- this branch used to
            # carry its own copy, which is how it ended up missing the coat
            # thickness, the rough G2/G1 and the chrominance floor the path
            # tracer had. Integrator bookkeeping (NEE per recycle depth, LVC
            # vertex storage, connect/merge, MIS carries, the
            # radiance-transport eta^2) stays here; only the material's own
            # behaviour moved out.
            var cw = coat_walk_begin(gn, wo, eff_alb, ior, coat_alpha, pcg)
            var cos_o = cw.cos_o

            # Coat's own glossy dielectric lobe NEE -- fired unconditionally
            # for a rough coat (independent of the reflect/transmit coin
            # flip below), matching shade_coated_diffuse's own rationale:
            # gating on the coin flip would double-count the interface
            # Fresnel term. Area lights skipped (LVC connect/merge already
            # covers them, same scope as every other NEE block in this
            # function).
            if is_rough_coat and cos_o > Float32(0):
                var (_cc_c, r_cc) = _scene_bounding_sphere(sd)
                var inv_scene_c = Float32(1.0) / max(_bdpt_n_lights(sd) * PI * r_cc * r_cc, Float32(1e-12))
                for li_c in range(_bdpt_simple_light_count(sd)):
                    var ls_ic = _bdpt_sample_simple_light(sd, li_c, hit.to_simd(), pcg)
                    # VCM's four-strategy balance share, not the path tracer's
                    # two-strategy power heuristic. Merging and t=1 reach this
                    # vertex too; taking weight 1 here double-counts, and this
                    # block runs ONLY for a rough coat.
                    var emis_c = inv_scene_c if li_c < Int(sd.distantLightCount) else Float32(0)
                    var pol_c = MisPolicy(True, mis_vm_weight_factor, dvcm_carry, dvc_carry,
                                          emis_c, Float32(0), False)
                    var w_ic = _nee_weight_coated_coat_lobe(ls_ic, ior, coat_alpha, gn, wo, pol_c)
                    total += _bdpt_nee_contribute(beta, spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, w_ic.r, w_ic.g, w_ic.b, wavelengths), ls_ic, hit, gn, cur_med_idx, sd, scratch, wavelengths)
                for inf_ic in range(Int(sd.infiniteLightCount)):
                    var ls_infc = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_ic], Point2f(pcg.next_float(), pcg.next_float()))
                    var pol_ic = MisPolicy(True, mis_vm_weight_factor, dvcm_carry, dvc_carry,
                                           ls_infc.pdf * inv_scene_c, Float32(0), False)
                    var w_infc = _nee_weight_coated_coat_lobe(ls_infc, ior, coat_alpha, gn, wo, pol_ic)
                    total += _bdpt_nee_contribute(beta, spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, w_infc.r, w_infc.g, w_infc.b, wavelengths), ls_infc, hit, gn, cur_med_idx, sd, scratch, wavelengths)

            coat_walk_enter(cw, pcg)

            if cw.event == COAT_ABSORB:
                return False   # coat total-internal-reflection-like case: terminate

            if cw.event == COAT_REFLECT:
                # Glossy reflection off the coat (rough => GGX lobe, smooth => mirror).
                var refl = cw.wi
                rd = vec3f(refl)
                ro = hit + rd*Float32(0.0002)
                if is_rough_coat:
                    # cw.beta carries the rough lobe's G2(wo,wi)/G1(wo)
                    # masking-shadowing weight. Achromatic by construction on
                    # this path (see CoatWalk.beta), hence the scalar multiply
                    # rather than a spectral upsample of a bare weight.
                    beta *= cw.beta.r
                    last_bsdf_pdf = cw.pdf
                    var v = _null_vertex()
                    v.pos = hit
                    v.normal = vec3f(gn_geo)
                    v.shading_normal = vec3f(gn)
                    v.beta = beta
                    v.alb = eff_alb
                    # coated_REFLECT, not coated_walk: this is the coat's own
                    # glossy bounce off the top interface, which never reaches
                    # the base. Stored as coated_walk it was re-evaluated by
                    # connect/merge through coat_eval_smooth -- the base
                    # transmission model, carrying the base's albedo -- for a
                    # path that never touched the base.
                    v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.coated_reflect
                    v.mat_idx = Int32(mat_idx)   # the coat evaluator reads ior from it
                    v.pdf_bwd = coat_alpha       # smooth-vs-rough, as ggx stores alpha
                    v.wo = vec3f(wo)
                    # REAL forward density and REAL carries, as the smooth
                    # exit vertex below already stores. These were
                    # `pdf_fwd = 1` and dVCM/dVC/dVM = 0 placeholders, valid
                    # only while lobe_scoped() still excluded a rough coat
                    # and _bdpt_merge_from_cache's scope gate therefore
                    # dropped this vertex. lobe_scoped now returns
                    # `c.is_surface` for EVERY coated_walk, rough included,
                    # so the gate stopped firing and this vertex began
                    # merging and connecting with fabricated weights: zero
                    # carries shrink the MIS denominator, so each strategy
                    # took a near-1 share and their sum over-counted.
                    # cw.pdf is the density actually sampled, which is what
                    # the dVCM recursion wants (SmallVCM does the same);
                    # the reverse half comes from the evaluator that
                    # merge/connect will themselves query at this vertex,
                    # so the recursion and the estimators agree.
                    v.pdf_fwd = cw.pdf
                    v.med_idx = cur_med_idx
                    v.wavelengths = wavelengths
                    v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                    if n_verts == 0: first_alb = eff_alb
                    n_verts += 1
                    # BUG (found 2026-09-15, chasing coateddiffuse-eta-probe
                    # rendering solid black under --vcm): every OTHER
                    # material branch in this function connects/merges its
                    # vertex against the light-path cache right after
                    # storing it (see the diffuse/conductor/hair/measured
                    # branches' identical `if path_len > 0:` block). This
                    # coateddiffuse vertex never did -- confirmed via direct
                    # instrumentation: the exit vertex below was reached
                    # with a perfectly valid nonzero walk_beta every time,
                    # but _connect was never once called for it. Since
                    # coateddiffuse also has no NEE path to an (unobstructed)
                    # area light -- only distant/point/sphere/infinite via
                    # _bdpt_simple_light_count, plus MNEE which only fires
                    # when the light is glass-obscured -- a scene lit purely
                    # by an area light through a coateddiffuse surface had
                    # NO way to receive any light at all.
                    # Merging queries the GLOBAL photon grid and does not use this
                    # pixel's own paired light path, so unlike the connect below it
                    # must NOT be gated on that path having stored anything. It was,
                    # and in a white furnace only ~32% of light paths hit the quad at
                    # all, so ~68% of pixels skipped merging entirely: the estimator
                    # delivered 0.109 against an analytic 0.5.
                    total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                    if path_len > 0:
                        total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)
                    # Propagate the carries THROUGH this bounce, as every
                    # scoped material does after it scatters. This used to
                    # fall through to the `dvcm_carry = 0` below, which left
                    # the whole REST of the path with no MIS state -- the
                    # smooth exit vertex's own comment records that giving a
                    # vertex real carries while the path downstream stays
                    # broken measures WORSE than leaving both alone, so the
                    # two have to move together.
                    var cos_out_cr = abs(dot(refl, gn))
                    var (_pf_cr, pdf_rev_cr) = _bdpt_vertex_pdfs(v, vec3f(refl), sd)
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                        dvcm_carry, dvc_carry, dvm_carry,
                        cos_out_cr / max(cw.pdf, Float32(1e-9)), cw.pdf, pdf_rev_cr,
                        mis_vc_weight_factor, mis_vm_weight_factor)
                else:
                    last_bsdf_pdf = Float32(-1)  # smooth mirror coat: delta, no MIS at destination
                    # Genuinely delta: dVCM resets, same convention as the
                    # dielectric branch and as volume scatter.
                    dvcm_carry = Float32(0)
                return True   # coat-reflect (rough): vertex already stored above, path continues

            # Transmitted into the coat: random-walk the base/coat-underside
            # layers. Entry attenuation already applied by coat_walk_enter.
            while cw.event == COAT_WALKING:
                if not coat_walk_at_base(cw, pcg):
                    break
                var walk_beta = cw.beta

                # coat_alpha (rough-coat G2/G1) and nee_is_sole_strategy=True
                # (this walk's exit ray always sets last_bsdf_pdf=0 below --
                # see _nee_weight_coated_diffuse_base's own docstring for
                # why the caller's continuation-ray convention decides this)
                # match shading.mojo's identical NEE calls exactly.
                # Per-iteration NEE is gone for every coat, not just smooth
                # ones: it sums the TIR series by REPETITION, a multi-vertex
                # model, while merging/t=1/the escape all model this vertex
                # once with coat_eval_smooth summing that series in closed
                # form. Two models of one vertex cannot partition unity. The
                # single-shot replacement is below, after the vertex exists.
                for li_b in range(0):
                    var ls_ib = _bdpt_sample_simple_light(sd, li_b, hit.to_simd(), pcg)
                    var w_ib = _nee_weight_coated_diffuse_base[True](ls_ib, eff_alb, ior, gn, coat_alpha)
                    total += _bdpt_nee_contribute(beta * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (walk_beta).r, (walk_beta).g, (walk_beta).b, wavelengths), spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, w_ib.r, w_ib.g, w_ib.b, wavelengths), ls_ib, hit, gn, cur_med_idx, sd, scratch, wavelengths)
                for inf_i in range(0):
                    var ls_inf = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float()))
                    # Now that a smooth coat walk is MIS-scoped, its NEE has
                    # to be weighted against the strategies that compete with
                    # it -- merging and t=1 -- like every other scoped kind.
                    # The reverse density is the same exit function of wo.
                    var (_c_ib, r_ib) = _scene_bounding_sphere(sd)
                    var pol_ib = MisPolicy(
                        not is_rough_coat, mis_vm_weight_factor, dvcm_carry, dvc_carry,
                        ls_inf.pdf / max(_bdpt_n_lights(sd) * PI * r_ib * r_ib, Float32(1e-12)),
                        bxdf_pdf_coated_exit(abs(dot(wo, gn)), ior), False)
                    # `[True]` means nee_is_sole_strategy -- the helper skips
                    # MIS and takes full weight. That WAS right while the coat
                    # walk never merged. A scoped smooth coat has competitors
                    # now, so it must take its balance share instead; a rough
                    # one is still unscoped and still sole.
                    # MEASURED, and left as-is deliberately: handing this NEE
                    # its balance share (the [False] instantiation with
                    # pol_ib) collapses the furnace to 0.6149 against an
                    # analytic 1.0, because the share it gives up is reserved
                    # for merging and merging does not deliver it. Same
                    # signature as the env case, where the cause turned out
                    # to be broken ESTIMATORS rather than weights -- so the
                    # next step is the isolation diagnostic on merging at a
                    # coated vertex, not another weight.
                    # A scoped smooth coat has competitors now (merging,
                    # t=1), so its NEE takes a balance share instead of full
                    # weight. A rough coat is still unscoped and still sole.
                    # A scoped smooth coat has competitors (merging, t=1), so
                    # its NEE takes a balance share. A rough coat is unscoped
                    # and still the sole strategy.
                    _ = pol_ib
                    var w_inf = _nee_weight_coated_diffuse_base[True](ls_inf, eff_alb, ior, gn, coat_alpha)
                    total += _bdpt_nee_contribute(beta * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (walk_beta).r, (walk_beta).g, (walk_beta).b, wavelengths), spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, w_inf.r, w_inf.g, w_inf.b, wavelengths), ls_inf, hit, gn, cur_med_idx, sd, scratch, wavelengths)

                # Task #161 follow-up (2026-07-13): MNEE for area lights
                # behind glass, extended to coateddiffuse's base layer --
                # see _bdpt_mnee_diffuse_area_light's docstring for why
                # this only ever fires for the glass-obscured case (no
                # double-count risk with the 4 NEE loops just above, which
                # only ever reach UNobstructed lights). Fired once per
                # recycling-walk iteration, matching those loops' own
                # per-iteration cadence -- hit/gn never change across
                # iterations (this walk is a fixed-position shading-space
                # recycling, not a real geometric random walk), only
                # walk_beta's attenuation does, so this models a genuinely
                # different bounce each iteration, not a repeated estimate
                # of the same one.
                total += _bdpt_mnee_diffuse_area_light(sd, hit, gn, eff_alb, beta * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (walk_beta).r, (walk_beta).g, (walk_beta).b, wavelengths), pcg, wavelengths, ior)
                # Sphere-shaped area lights are deliberately NOT MNEE'd
                # from the coat's base: _bdpt_mnee_sphere_light no longer
                # applies coat_t at all (see its docstring's "GPU codegen
                # bug" section), so calling it with a non-1.0 ior would skip
                # the coat's Fresnel attenuation and overestimate light
                # through the coat. Re-enable by mirroring the plain-diffuse
                # branch's unrolled call site once coat_t has a real fix.

                # One base bounce + the attempt to leave the coat -- shared.
                coat_walk_scatter(cw, pcg)

            var exited = cw.event == COAT_EXIT
            var exit_dir = cw.wi
            if not exited:
                return False   # coat recycling walk absorbed (RR-killed or ran out of depth)

            rd = vec3f(exit_dir)
            ro = hit + rd*Float32(0.0002)
            # A smooth coat's exit density is known (bxdf_pdf_coated_exit),
            # so this vertex gets REAL carries and joins MIS instead of the
            # placeholder zeros it used to store. Rough coats keep the old
            # NEE-only behaviour -- no closed form for their exit.
            var cos_x_c = abs(dot(exit_dir, gn))
            var smooth_coat = coat_alpha <= Float32(0.001)
            var pdf_x_c = bxdf_pdf_coated_exit(cos_x_c, ior)
            last_bsdf_pdf = pdf_x_c
            # 1/eta^2: the exit ray leaves the dense coat for air, so its
            # radiance is compressed by the squared IOR ratio -- see
            # shading.mojo's twin of this line and
            # _nee_weight_coated_diffuse_base, which carries the NEE side's
            # own copy. Missing on both, it was worth eta^2 (2.3x at 1.5).
            # No 1/eta^2 on the SAMPLED exit ray -- the exit refraction's
            # eta^2 Jacobian cancels it (see shading.mojo's twin of this line
            # for the sweep that settled it). coat_eval_smooth keeps its
            # 1/eta^2 because it evaluates a GIVEN direction.
            # The vertex's beta is the throughput arriving BEFORE the coat,
            # because a connection or a merge at this vertex supplies the coat
            # transport itself through the coat-exit lobe. Storing the
            # post-walk beta applies the coat TWICE -- and every one of those
            # factors is < 1 (entry Fresnel, Beer-Lambert, 1/eta^2 = 0.44 at
            # eta 1.5), so the double application reads as far too DARK, not
            # too bright. The continuing ray still takes the post-walk beta.
            var beta_pre_coat = beta
            beta *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (cw.beta).r, (cw.beta).g, (cw.beta).b, wavelengths)
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn_geo)
            v.shading_normal = vec3f(gn)
            v.beta = beta_pre_coat
            v.alb = eff_alb
            v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.coated_walk
            v.mat_idx = Int32(mat_idx)   # the coat evaluator reads ior from it
            v.pdf_bwd = coat_alpha       # smooth-vs-rough, as ggx stores alpha
            v.wo = vec3f(wo)
            v.pdf_fwd = pdf_x_c
            v.med_idx = cur_med_idx
            v.wavelengths = wavelengths
            # Real carries. The arrival half was already computed for this
            # hit at the top of the branch and then thrown away; a smooth
            # coat now keeps it. A rough coat has no exit density, so it
            # stays at zero AND out of scope -- the two must agree, because
            # scope without carries is the documented volume-mis failure:
            # MIS reserves share that no strategy then delivers.
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            if n_verts == 0: first_alb = eff_alb
            n_verts += 1
            # BUG (found 2026-09-15): same missing connect/merge call as the
            # coat-reflect vertex above -- see that site's comment for the
            # full story. This exit vertex is the ONE both smooth and rough
            # coats always reach, so this was THE fix for
            # coateddiffuse-eta-probe.pbrt rendering solid black under
            # --vcm (a coateddiffuse surface lit only by an unobstructed
            # area light had no path to receive any light at all).
            # Merging queries the GLOBAL photon grid and does not use this
            # pixel's own paired light path, so unlike the connect below it
            # must NOT be gated on that path having stored anything. It was,
            # and in a white furnace only ~32% of light paths hit the quad at
            # all, so ~68% of pixels skipped merging entirely: the estimator
            # delivered 0.109 against an analytic 0.5.
            total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
            if path_len > 0:
                total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)
            if True:
                # Simple lights (distant/point/sphere) single-shot too. The
                # sun is delta: its NEE cannot be found by BSDF sampling, but
                # merging and t=1 still compete for its photons, so it takes
                # a balance share rather than weight 1.
                for li_s in range(_bdpt_simple_light_count(sd)):
                    var ls_d = _bdpt_sample_simple_light(sd, li_s, hit.to_simd(), pcg)
                    var cos_d = dot(gn, ls_d.wi)
                    if ls_d.valid and cos_d > Float32(0):
                        var le_d = _lobe_eval[want_pdfs=True](v, ls_d.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                        var (_cd, r_d) = _scene_bounding_sphere(sd)
                        var emis_d = (Float32(1.0) / max(_bdpt_n_lights(sd) * PI * r_d * r_d, Float32(1e-12))
                                      if li_s < Int(sd.distantLightCount) else Float32(0))
                        var pol_d = MisPolicy(le_d.scoped, mis_vm_weight_factor, dvcm_carry, dvc_carry, emis_d, le_d.pdf_rev, False)
                        # The SAME evaluator the infinite-light shot and
                        # merging use. _nee_weight_coated_diffuse_base is the
                        # single-scatter form and omits the TIR series that
                        # coat_eval_smooth sums -- using it here would put two
                        # models of this vertex back in, which is the very
                        # thing the single-shot rewrite removed.
                        var mis_d = (nee_mis_weight(pol_d, ls_d.pdf, le_d.pdf_fwd, cos_d)
                                     if not ls_d.is_delta
                                     else nee_mis_weight(pol_d, Float32(1.0), Float32(0.0), cos_d))
                        var inv_pd = Float32(1.0) if ls_d.is_delta else (Float32(1.0) / ls_d.pdf)
                        var li_d = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, ls_d.Li.r, ls_d.Li.g, ls_d.Li.b, wavelengths)
                        total += _bdpt_nee_contribute(beta_pre_coat, le_d.f_cos * li_d * (mis_d * inv_pd), ls_d, hit, gn, cur_med_idx, sd, scratch, wavelengths)
                for inf_s in range(Int(sd.infiniteLightCount)):
                    var ls_s = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_s], Point2f(pcg.next_float(), pcg.next_float()))
                    var cos_s_c = dot(gn, ls_s.wi)
                    if ls_s.valid and cos_s_c > Float32(0) and ls_s.pdf > Float32(0):
                        var le_s = _lobe_eval[want_pdfs=True](v, ls_s.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                        var (_cc, r_s) = _scene_bounding_sphere(sd)
                        var mis_s = vcm_env_nee_weight(le_s.pdf_fwd, le_s.pdf_rev, ls_s.pdf, ls_s.pdf / max(_bdpt_n_lights(sd) * PI * r_s * r_s, Float32(1e-12)), cos_s_c, mis_vm_weight_factor, dvcm_carry, dvc_carry)
                        var li_s = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, ls_s.Li.r, ls_s.Li.g, ls_s.Li.b, wavelengths)
                        total += _bdpt_nee_contribute(beta_pre_coat, le_s.f_cos * li_s * (mis_s / ls_s.pdf), ls_s, hit, gn, cur_med_idx, sd, scratch, wavelengths)
            # Propagate the carries THROUGH the coat, exactly as every other
            # material does after it scatters. This used to be
            # `dvcm_carry = 0`, which left the whole REST of the path with no
            # MIS state -- so an earlier attempt that gave this vertex real
            # carries still measured worse, because every downstream vertex
            # was still broken. The vertex and the path have to be fixed
            # together or neither measurement means anything.
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                dvcm_carry, dvc_carry, dvm_carry,
                cos_x_c / max(pdf_x_c, Float32(1e-9)), pdf_x_c,
                bxdf_pdf_coated_exit(abs(dot(wo, gn)), ior),
                mis_vc_weight_factor, mis_vm_weight_factor)

        elif mat.type == MatKind.conductor or mat.type == MatKind.coated_conductor:
            var gn_c = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            # Bump/normal maps -- see the diffuse branch.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_c_geo = gn_c
            gn_c = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn_c, gn_c, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=Vec3f(frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=Vec3f(frm_c.y.x, frm_c.y.y, frm_c.y.z),
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
                return False   # conductor/coated_conductor sample invalid
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
                v.normal = vec3f(gn_c_geo)
                v.shading_normal = vec3f(gn_c)
                v.beta = beta
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.ggx
                v.pdf_bwd = alpha_c
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                if n_verts == 0: first_alb = mat.albedo
                n_verts += 1
                # Merging queries the GLOBAL photon grid and does not use this
                # pixel's own paired light path, so unlike the connect below it
                # must NOT be gated on that path having stored anything. It was,
                # and in a white furnace only ~32% of light paths hit the quad at
                # all, so ~68% of pixels skipped merging entirely: the estimator
                # delivered 0.109 against an analytic 0.5.
                total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                if path_len > 0:
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
                for li_cc in range(_bdpt_simple_light_count(sd)):
                    var ls_icc = _bdpt_sample_simple_light(sd, li_cc, hit.to_simd(), pcg)
                    var w_icc = _nee_weight_simple_spectral(ls_icc, LobeKind.ggx, mat.albedo, alpha_c, gn_c, wo_c, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs))
                    total += _bdpt_nee_contribute(beta, w_icc, ls_icc, hit, gn_c, cur_med_idx, sd, scratch, wavelengths)
                for inf_ic in range(Int(sd.infiniteLightCount)):
                    var ls_ec = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_ic], Point2f(pcg.next_float(), pcg.next_float()))
                    # One policy expression, and `scoped` decides: a kind with real
                    # densities gets VCM's balance weight over all four strategies, a
                    # kind without keeps the path tracer's two-strategy heuristic.
                    var (_c_ec, r_ec) = _scene_bounding_sphere(sd)
                    var le_ec = _lobe_eval[want_pdfs=True](v, ls_ec.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                    var pol_ec = MisPolicy(le_ec.scoped, mis_vm_weight_factor, dvcm_carry, dvc_carry,
                                          ls_ec.pdf / max(_bdpt_n_lights(sd) * PI * r_ec * r_ec, Float32(1e-12)), le_ec.pdf_rev, False)
                    var w_ec = _nee_weight_simple_spectral(ls_ec, LobeKind.ggx, mat.albedo, alpha_c, gn_c, wo_c, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs), pol_ec)
                    total += _bdpt_nee_contribute(beta, w_ec, ls_ec, hit, gn_c, cur_med_idx, sd, scratch, wavelengths)

            beta *= spec_refl_unbounded(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (bs_c.f).r, (bs_c.f).g, (bs_c.f).b, wavelengths)
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
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                        dvcm_carry, dvc_carry, dvm_carry, inv_pdf_c, bsdf_dir_pdf_w_c, bsdf_rev_pdf_w_c,
                        mis_vc_weight_factor, mis_vm_weight_factor)
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
            # VCM: hair DOES have a real standalone pdf (_hair_eval_lobes,
            # same one _nee_weight_hair already trusts for NEE MIS) -- see
            # _bdpt_vertex_pdfs' mat_kind=2 branch, which closes the
            # connect/merge-time weighting. This cos_fix is the same
            # distance²/area-measure correction diffuse/conductor apply.
            var cos_fix_h = abs(dot(-ray_dir, hc.geo_normal))
            if cos_fix_h > Float32(1e-6):
                dvc_carry /= cos_fix_h
                dvm_carry /= cos_fix_h
            var v_h = _null_vertex()
            v_h.pos = hit
            v_h.normal = vec3f(hc.geo_normal)
            v_h.shading_normal = vec3f(hc.geo_normal)
            v_h.beta = beta
            v_h.alb = mat.albedo
            v_h.is_surface = Int32(1); v_h.is_delta = Int32(0); v_h.mat_kind = LobeKind.hair
            v_h.wo = vec3f(wo_h)
            v_h.mat_idx = Int32(mat_idx)
            v_h.hair_curve_idx = Int32(curve_idx_h)
            v_h.hair_h = inter.u
            v_h.hair_v = inter.v
            v_h.pdf_fwd = Float32(1)
            v_h.med_idx = cur_med_idx
            v_h.wavelengths = wavelengths
            v_h.dVCM = dvcm_carry; v_h.dVC = dvc_carry; v_h.dVM = dvm_carry
            if n_verts == 0: first_alb = mat.albedo
            n_verts += 1
            # Merging queries the GLOBAL photon grid and does not use this
            # pixel's own paired light path, so unlike the connect below it
            # must NOT be gated on that path having stored anything. It was,
            # and in a white furnace only ~32% of light paths hit the quad at
            # all, so ~68% of pixels skipped merging entirely: the estimator
            # delivered 0.109 against an analytic 0.5.
            total += _bdpt_merge_from_cache(v_h, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
            if path_len > 0:
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
            for li_h in range(_bdpt_simple_light_count(sd)):
                var ls_ih = _bdpt_sample_simple_light(sd, li_h, hit.to_simd(), pcg)
                var w_ih = _nee_weight_hair(ls_ih, hc)
                total += _bdpt_nee_contribute(beta, spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, w_ih.r, w_ih.g, w_ih.b, wavelengths), ls_ih, hit, hc.geo_normal, cur_med_idx, sd, scratch, wavelengths, hair_eps)
            for inf_ih in range(Int(sd.infiniteLightCount)):
                var ls_eh = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_ih], Point2f(pcg.next_float(), pcg.next_float()))
                # One policy expression, and `scoped` decides: a kind with real
                # densities gets VCM's balance weight over all four strategies, a
                # kind without keeps the path tracer's two-strategy heuristic.
                var (_c_eh, r_eh) = _scene_bounding_sphere(sd)
                var le_eh = _lobe_eval[want_pdfs=True](v_h, ls_eh.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
                var pol_eh = MisPolicy(le_eh.scoped, mis_vm_weight_factor, dvcm_carry, dvc_carry,
                                      ls_eh.pdf / max(_bdpt_n_lights(sd) * PI * r_eh * r_eh, Float32(1e-12)), le_eh.pdf_rev, False)
                var w_eh = _nee_weight_hair(ls_eh, hc, pol_eh)
                total += _bdpt_nee_contribute(beta, spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, w_eh.r, w_eh.g, w_eh.b, wavelengths), ls_eh, hit, hc.geo_normal, cur_med_idx, sd, scratch, wavelengths, hair_eps)

            var (wi_hs, f_hs, pdf_hs, cos_ti_hs2) = _hair_sample_dir(hc, pcg)
            beta *= spec_refl_unbounded(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (f_hs / pdf_hs).r, (f_hs / pdf_hs).g, (f_hs / pdf_hs).b, wavelengths)
            rd = vec3f(wi_hs)
            var hsign = Float32(1) if dot(wi_hs, hc.geo_normal) >= Float32(0) else Float32(-1)
            ro = hit + vec3f(hc.geo_normal) * hair_eps * hsign
            last_bsdf_pdf = pdf_hs * cos_ti_hs2  # solid-angle pdf (pdf_hs already has /cos_ti baked in)
            var cos_theta_out_h = abs(dot(wi_hs, hc.geo_normal))
            # VCM: real non-specular recursive update, mirroring conductor's
            # Stage 2d treatment -- see _bdpt_trace_light_path's matching
            # branch for the general (cosThetaOut/bsdfDirPdfW) form.
            # bsdf_dir_pdf_w_h is exactly last_bsdf_pdf, just computed
            # already above; the REVERSE pdf needs hair's lobes
            # re-evaluated with wo_h/wi_hs swapped -- same "second
            # precompute with wo=dir_to_other" pattern _bdpt_vertex_pdfs'
            # own hair branch uses, since hair's "wo" is baked into the
            # precomputed HairLobeConstants rather than passed per-call.
            var bsdf_dir_pdf_w_h = last_bsdf_pdf
            if bsdf_dir_pdf_w_h > Float32(1e-8):
                var hc_rev_h = _hair_precompute(mat, sd.curves, curve_idx_h, inter.v, inter.u, wi_hs)
                var (cos_ti_rev_h, _, pdf_oc_rev_h) = _hair_eval_lobes(
                    wo_h, hc_rev_h.tangent, hc_rev_h.b_perp, hc_rev_h.n_perp, hc_rev_h.phi_o,
                    hc_rev_h.dphi0, hc_rev_h.dphi1, hc_rev_h.dphi2,
                    hc_rev_h.cos_tp0_o, hc_rev_h.sin_tp0_o, hc_rev_h.cos_tp1_o, hc_rev_h.sin_tp1_o, hc_rev_h.cos_tp2_o, hc_rev_h.sin_tp2_o,
                    hc_rev_h.cos_theta_o, hc_rev_h.sin_theta_o, hc_rev_h.inv_vm0, hc_rev_h.inv_vm1, hc_rev_h.inv_vm2, hc_rev_h.mp_c0, hc_rev_h.mp_c1, hc_rev_h.mp_c2, hc_rev_h.s,
                    hc_rev_h.A0, hc_rev_h.A1, hc_rev_h.A2, hc_rev_h.A3, hc_rev_h.lum0, hc_rev_h.lum1, hc_rev_h.lum2, hc_rev_h.lum3, hc_rev_h.total_lum,
                )
                var bsdf_rev_pdf_w_h = cos_ti_rev_h * pdf_oc_rev_h
                var inv_pdf_h = cos_theta_out_h / bsdf_dir_pdf_w_h
                (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                    dvcm_carry, dvc_carry, dvm_carry, inv_pdf_h, bsdf_dir_pdf_w_h, bsdf_rev_pdf_w_h,
                    mis_vc_weight_factor, mis_vm_weight_factor)
            else:
                dvcm_carry = Float32(0)
                dvc_carry = Float32(0)
                dvm_carry = Float32(0)

        elif mat.type == MatKind.measured:
            # Tabulated Dupuy & Jakob MeasuredBxDF -- the real algorithm
            # (measured_bxdf_eval.mojo), not an approximation, via the same
            # shared bxdf.mojo/measured_bxdf_eval.mojo interface
            # shading.mojo's shade_measured already uses. Isotropic only (see
            # the loader's scope note), so an arbitrary Frisvad tangent frame
            # is fine -- no UV alignment needed, same reasoning as conductor.
            var gn_m = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn_m, ray_dir) > Float32(0): gn_m = gn_m * Float32(-1)
            # Bump/normal maps -- see the diffuse branch.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_m_geo = gn_m
            gn_m = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn_m, gn_m, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            if mat.measured_idx < Int32(0):
                # Load failure fallback (see material_builder.mojo) -- matches
                # shading.mojo's shade_measured: stop this path rather than
                # dereferencing a nonexistent measuredBrdfs entry.
                return False   # measured: no tabulated BRDF for this material
            var wo_m = (-rd).to_simd()
            var frm_m = Frame.from_z(Vec3f(gn_m[0], gn_m[1], gn_m[2]))
            var tangent_m = Vec3f(frm_m.x.x, frm_m.x.y, frm_m.x.z)
            var bitangent_m = Vec3f(frm_m.y.x, frm_m.y.y, frm_m.y.z)
            var mb = sd.measuredBrdfs[unsafe_offset=Int(mat.measured_idx)]

            # VCM Stage 2b: measured BxDF has a real standalone pdf -- in
            # MIS scope this pass, same real treatment as diffuse.
            var cos_fix_m = abs(dot(-ray_dir, gn_m))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix_m)

            var v_m = _null_vertex()
            v_m.pos = hit
            v_m.normal = vec3f(gn_m_geo)
            v_m.shading_normal = vec3f(gn_m)
            v_m.beta = beta
            v_m.alb = mat.albedo
            v_m.is_surface = Int32(1); v_m.is_delta = Int32(0); v_m.mat_kind = LobeKind.measured
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
                # always True here; kept for symmetry with the volume
                # branch's matching guard against merge+connect double-
                # counting for any future out-of-scope kind (only volume
                # and delta dielectric remain out of scope now that hair
                # is closed too).
                if _bdpt_vertex_mis_scoped(v_m):
                    total += _bdpt_merge_from_cache(v_m, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                total += _bdpt_connect_to_cache(v_m, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)

            # Distant/point/sphere/infinite NEE, via the shared Light
            # interface + BxDF interface (_nee_weight_measured) +
            # _bdpt_nee_contribute glue -- same pattern as every other
            # material branch above.
            for li_m in range(_bdpt_simple_light_count(sd)):
                var ls_im = _bdpt_sample_simple_light(sd, li_m, hit.to_simd(), pcg)
                var w_im = _nee_weight_measured(ls_im, mb, tangent_m, bitangent_m, gn_m, wo_m, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                total += _bdpt_nee_contribute(beta, w_im, ls_im, hit, gn_m, cur_med_idx, sd, scratch, wavelengths)
            for inf_im in range(Int(sd.infiniteLightCount)):
                var ls_em = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_im], Point2f(pcg.next_float(), pcg.next_float()))
                var w_em = _nee_weight_measured(ls_em, mb, tangent_m, bitangent_m, gn_m, wo_m, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                total += _bdpt_nee_contribute(beta, w_em, ls_em, hit, gn_m, cur_med_idx, sd, scratch, wavelengths)

            var wo_l_m = Vec3f(dot(wo_m, tangent_m), dot(wo_m, bitangent_m), dot(wo_m, gn_m))
            var um1 = pcg.next_float(); var um2 = pcg.next_float()
            var (wi_l_m, f_m, pdf_m, valid_m) = bxdf_sample_measured(mb, wo_l_m, um1, um2, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
            if not valid_m or pdf_m <= Float32(0):
                return False   # measured: invalid sample
            var wi_m = tangent_m * wi_l_m[0] + bitangent_m * wi_l_m[1] + gn_m * wi_l_m[2]
            var wilen_m = dot(wi_m, wi_m)
            if wilen_m > Float32(0):
                wi_m = wi_m * (Float32(1.0) / sqrt(wilen_m))
            var cos_wi_m = dot(wi_m, gn_m)
            if cos_wi_m <= Float32(0):
                return False   # measured: sampled direction below the surface
            # f_m is already spectral -- no RGB round trip (see
            # bxdf_sample_measured's docstring).
            beta *= f_m * (cos_wi_m / pdf_m)
            rd = vec3f(wi_m)
            ro = hit + rd*Float32(0.0002)
            last_bsdf_pdf = pdf_m
            # VCM Stage 2b: recursive continuation, real forward/reverse pdf
            # from bxdf_pdf_measured -- see _bdpt_trace_light_path's
            # matching measured-branch comment (reverse-pdf convention
            # ASSUMED, not independently verified).
            var pdf_rev_m = bxdf_pdf_measured(mb, wi_l_m, wo_l_m)
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                dvcm_carry, dvc_carry, dvm_carry, (cos_wi_m / pdf_m), pdf_m, pdf_rev_m,
                mis_vc_weight_factor, mis_vm_weight_factor)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var did_bssrdf_hop = False
            # ── Subsurface boundary: a BSSRDF hop to a real EXIT VERTEX ────
            # The camera arrives at the entry x_i, the exit x_o is sampled from
            # the diffusion profile (shared with the light subpath, see
            # _bdpt_sample_bssrdf_exit), and x_o then becomes an ordinary
            # connectible vertex: merge, connect, direct lighting, continue.
            # Its BSDF is the exit lobe Ft(cos)/pi (LobeKind.bssrdf), whose pdfs are
            # exactly Lambertian. The MIS carries across the hop follow the
            # rule machine-checked in Scenes/vcm_bssrdf_mis_derivation.py.
            # The entry itself is never connectible.
            if mat.sss_boundary != Int8(0) and Int(cur_med_idx) < 0 and has_med:
                var med_in = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if med_in >= Int32(0) and Int(med_in) < Int(sd.mediumCount):
                    var gn_e = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                    if dot(gn_e, ray_dir) > Float32(0): gn_e = gn_e * Float32(-1)
                    var cos_e = abs(dot(-ray_dir, gn_e))
                    # Arrival at the entry, as at any surface vertex.
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                        dvcm_carry, dvc_carry, dvm_carry, cos_e)
                    var eta_e = mat.albedo.r
                    var ex = _bdpt_sample_bssrdf_exit(sd, Int(med_in), hit, gn_e, cos_e, eta_e, pcg)
                    if not ex.ok:
                        return False
                    beta *= spec_refl_unbounded(
                        sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                        sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                        ex.weight.r, ex.weight.g, ex.weight.b, wavelengths)
                    var (h0, h1, h2) = bssrdf_hop_carries(
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_e * INV_PI, mis_vc_weight_factor)
                    dvcm_carry = h0
                    dvc_carry = h1
                    dvm_carry = h2
                    var x_o = ex.x_o
                    var n_o = ex.n_o
                    var v = _null_vertex()
                    v.pos = x_o
                    v.normal = vec3f(n_o)
                    v.shading_normal = vec3f(n_o)
                    v.beta = beta
                    v.alb = RGB(Float32(1))
                    v.is_surface = Int32(1); v.is_delta = Int32(0)
                    v.mat_kind = LobeKind.bssrdf
                    v.pdf_fwd = ex.p_area      # reverse density toward x_i (hop is symmetric)
                    v.pdf_bwd = eta_e
                    v.wo = vec3f(n_o)          # unused by LobeKind.bssrdf
                    v.med_idx = cur_med_idx
                    v.wavelengths = wavelengths
                    v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                    if n_verts == 0: first_alb = mat.sss_mean_refl
                    n_verts += 1
                    # Merging queries the GLOBAL photon grid and does not use this
                    # pixel's own paired light path, so unlike the connect below it
                    # must NOT be gated on that path having stored anything. It was,
                    # and in a white furnace only ~32% of light paths hit the quad at
                    # all, so ~68% of pixels skipped merging entirely: the estimator
                    # delivered 0.109 against an analytic 0.5.
                    total += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor)
                    if path_len > 0:
                        if defer_shadow_rays:
                            _bdpt_connect_to_cache_deferred(v, sd, lvc, lp_idx, path_len, mis_vm_weight_factor, shadow_rays, shadow_pending, shadow_valid, shadow_seg_med)
                        else:
                            total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor)
                    # Direct lighting at the exit: the diffuse vertex's NEE with
                    # the exit lobe's Fresnel factor toward each light.
                    for li_x in range(_bdpt_simple_light_count(sd)):
                        var ls_x = _bdpt_sample_simple_light(sd, li_x, x_o.to_simd(), pcg)
                        var w_x = _nee_weight_simple_spectral(ls_x, LobeKind.lambertian, RGB(Float32(1)), Float32(0), n_o, n_o, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs))
                        w_x = w_x * bssrdf_exit_ft(dot(n_o, ls_x.wi), eta_e)
                        total += _bdpt_nee_contribute(beta, w_x, ls_x, x_o, n_o, cur_med_idx, sd, scratch, wavelengths)
                    for inf_x in range(Int(sd.infiniteLightCount)):
                        var ls_xe = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_x], Point2f(pcg.next_float(), pcg.next_float()))
                        var w_xe = _nee_weight_simple_spectral(ls_xe, LobeKind.lambertian, RGB(Float32(1)), Float32(0), n_o, n_o, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs))
                        w_xe = w_xe * bssrdf_exit_ft(dot(n_o, ls_xe.wi), eta_e)
                        total += _bdpt_nee_contribute(beta, w_xe, ls_xe, x_o, n_o, cur_med_idx, sd, scratch, wavelengths)
                    # Continue with the exit lobe: cosine-sampled, weight Ft(cos_out).
                    var ux1 = pcg.next_float(); var ux2 = pcg.next_float()
                    rd = vec3f(_cosine_hemisphere_sample(n_o, ux1, ux2))
                    ro = x_o + rd * Float32(0.0002)
                    var cos_out_x = abs(dot(rd.to_simd(), n_o))
                    last_bsdf_pdf = cos_out_x * INV_PI
                    beta *= SpectralSample(bssrdf_exit_ft(cos_out_x, eta_e))
                    var (c0, c1, c2) = bssrdf_exit_scatter_carries(
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_out_x,
                        mis_vc_weight_factor, mis_vm_weight_factor)
                    dvcm_carry = c0
                    dvc_carry = c1
                    dvm_carry = c2
                    n_bounces += 1
                    did_bssrdf_hop = True
            # Ordinary specular boundary, only when no hop was taken.
            if not did_bssrdf_hop:
                var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                # Bump/normal maps. barcelona's water is a `dielectric` with a
                # "texture displacement", so this is THE branch the corpus
                # cares about most -- and VCM refracted through a perfectly
                # flat sheet while the path tracer refracted through a
                # rippled one. `orient_to` is the RAW winding normal, NOT a
                # face-forwarded one: _dielectric_bounce decides entering vs
                # exiting from dot(ray_dir, n) < 0, and flipping the
                # perturbed normal toward the ray would make that test
                # tautological and resurrect the 1/eta^4 transmission loss
                # (same rule as shading.mojo's dielectric site).
                gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                    gn, gn, ray_dir,
                    _camera_approx_footprint(hit, cam_pos, px_scale),
                    sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
                var (new_dir, new_org, radiance_scale, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                    ray_dir, hit.to_simd(), gn, mat.albedo.r, n_bounces == 0 and Int(cur_med_idx) < 0, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric)
                current_dielectric_ior = new_cur_ior
                previous_dielectric_ior = new_prev_ior
                n_bounces += 1
                last_bsdf_pdf = Float32(-1)  # delta bounce: no infinite-light NEE done here
                # Specular vertex: no BSDF record needed, just track throughput.
                # Camera path (Radiance mode): apply the non-symmetric-scattering
                # correction (see _dielectric_bounce's docstring).
                beta *= radiance_scale
                if has_med:
                    var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
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
                var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)
            mis_null_dist += t_hit + Float32(0.0002)   # see VCMCameraPathState_C.mis_null_dist
            # VCM Stage 2b: pure pass-through, carry unchanged (see
            # _bdpt_trace_light_path's matching interface-branch comment).

        else:
            return False   # unhandled material type


        return True   # bounce processed normally, path continues

# ── Trace one light subpath, storing its vertices into the shared cache ─────


@fieldwise_init
struct VCMLightPathState_C(TrivialRegisterPassable):
    """Task #163 stage 4: persistent per-light-path state carried across
    separate wavefront-staged GPU kernel launches (`_bdpt_light_path_init_gpu`
    then one `_bdpt_light_path_bounce_gpu` call per bounce), the light-path
    counterpart to gpu.mojo's `PathState_C` for the plain wavefront path
    tracer. `active=0` means the path is done (produced by
    `_null_light_path_state()` or by `_bdpt_light_path_bounce` returning
    False) -- the host loop stops calling the bounce kernel for a lane once
    its `active` flag drops to 0, matching PathState_C's own convention."""
    var ro: Point3f
    var rd: Vec3f
    var flux: SpectralSample
    var dvcm: Float32
    var dvc: Float32
    var dvm: Float32
    var is_finite_origin: Int8
    var cur_med_idx: Int32
    var n_lbounces: Int32
    var n_verts: Int32
    var active: Int8
    var pcg_state: UInt64
    var pcg_inc: UInt64
    var wl0: Float32
    var wl1: Float32
    var wl2: Float32
    var wl3: Float32
    var wl_pdf: Float32
    # Touching-dielectric IOR depth-2 stack for _dielectric_bounce (see that
    # function's docstring, sppm.mojo) -- same role and convention as
    # VCMCameraPathState_C's matching fields. Both start at vacuum (1.0).
    var current_dielectric_ior: Float32
    var previous_dielectric_ior: Float32

def _null_light_path_state() -> VCMLightPathState_C:
    return VCMLightPathState_C(
        Point3f(Float32(0), Float32(0), Float32(0)),
        Vec3f(Float32(0), Float32(0), Float32(0)),
        SpectralSample(Float32(0)),
        Float32(0), Float32(0), Float32(0), Int8(0),
        Int32(-1), Int32(0), Int32(0), Int8(0),
        UInt64(0), UInt64(0),
        Float32(0), Float32(0), Float32(0), Float32(0), Float32(0),
        Float32(1.0), Float32(1.0),   # current_dielectric_ior, previous_dielectric_ior (vacuum)
    )

def _bdpt_light_path_init[use_gpu: Bool](
    ref sd: SceneDescriptor2_C,
    mut pcg: PCG32,
    default_emit_med: Int32,
    lp_idx: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    pass_wl: SampledWavelengths,
) -> VCMLightPathState_C:
    """Task #163 stage 4: light-emission setup half of
    `_bdpt_trace_light_path` (bdpt.mojo:1734-1880), split out to seed a
    `VCMLightPathState_C` for the wavefront-staged bounce loop instead of
    falling straight into an inline `for` loop. Byte-for-byte copy of that
    function's pre-loop body -- see its own docstring/VCM Stage 2b comments
    for the MIS derivation, not repeated here. The only changes are the two
    bare `return` sites (no lights in the scene; zero-pdf infinite-light dir
    sample) now returning `_null_light_path_state()` (active=0) instead, and
    the end of the function packaging the locals into a returned state
    instead of continuing into a bounce loop."""
    var n_area = Int(sd.areaLightCount)
    var n_distant = Int(sd.distantLightCount)
    var n_infinite = Int(sd.infiniteLightCount)
    var n_point = Int(sd.pointLightCount)
    var n_lights = n_area + n_distant + n_infinite + n_point
    lvc_path_len[unsafe_offset=lp_idx] = Int32(0)
    if n_lights == 0:
        return _null_light_path_state()   # no lights in the scene

    var ro: Point3f
    var rd: Vec3f
    # A light subpath's flux is EMISSION, so it crosses into the spectral
    # domain through the illuminant curve -- the same boundary the camera
    # subpath crosses at an emitter hit.
    var flux: SpectralSample
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
    # This pass's shared hero wavelengths (see _bdpt_camera_path_init's
    # comment for why every subpath in a pass must agree on them).
    var wavelengths = pass_wl

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
        lv0_vert.shading_normal = vec3f(ln)
        lv0_vert.beta = SpectralSample(area_weight)
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
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, wavelengths) * (area_weight * PI)
        ro = point3f(lp) + vec3f(ln)*Float32(0.0001)
        rd = vec3f(pdir)
        n_verts = 1  # vertex 0 is the light point itself
    elif light_pick < n_area + n_distant:
        var dl = sd.distantLights[unsafe_offset=light_pick - n_area]
        var (center, radius) = _scene_bounding_sphere(sd)
        var dir = Vec3f(dl.direction.x, dl.direction.y, dl.direction.z)
        var disk_pt = _sample_disk_perpendicular(dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        # Phi_light = emission(irradiance) × disk_area; p_i = 1/n_lights;
        # pdf_pos = 1/disk_area; pdf_dir = 1 (delta) → flux = emission × disk_area × n_lights.
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, dl.emission.r, dl.emission.g, dl.emission.b, wavelengths) * (Float32(n_lights) * PI * radius * radius)
        ro = disk_pt
        rd = dir
        n_verts = 0  # no finite light point to store as a cache vertex
        # Real MIS origin for a DISTANT light -- the third instance of the
        # zero-carries bug: c1f2e10f fixed the infinite branch and left this
        # one at 0 ("scoped simplification"). Zero carries collapse every
        # merge-with and t=1 weight for a sun photon to near-full credit while
        # the camera's delta-light NEE ALSO takes full weight: classroom (sun +
        # env) read 2x pbrt with merging on, 0.985x with it off, and none of
        # the exact scenes could see it because none has a distant light.
        # SmallVCM's directional case: directPdfW = 1 (delta), emissionPdfW =
        # the disk's area pdf, so dVCM = diskArea and dVC = dVM = 0 (a delta
        # light has no cosine and no BSDF-sampleable direction). n_lights
        # multiplies dVCM because the CAMERA's NEE loops every light with no
        # pick while this path picked one with probability 1/n_lights.
        dvcm_carry = PI * radius * radius * Float32(n_lights)
        dvc_carry = Float32(0)
        dvm_carry = Float32(0)
    elif light_pick < n_area + n_distant + n_infinite:
        var il = sd.infiniteLights[unsafe_offset=light_pick - n_area - n_distant]
        var (center, radius) = _scene_bounding_sphere(sd)
        # _sample_infinite_light_dir returns env_dir in the NEE convention
        # ("direction FROM a shading point TOWARD the light" — same as
        # shading.mojo's _nee_infinite_light usage). A photon leaving the
        # light travels the opposite way, arriving FROM that direction
        # INTO the scene — negate it for the emitted ray/disk placement.
        var (env_dir, env_rgb, pdf_dir) = _sample_infinite_light_dir(il, Point2f(pcg.next_float(), pcg.next_float()))
        if pdf_dir <= Float32(0):
            return _null_light_path_state()   # infinite-light dir sample had zero pdf
        var emit_dir = -env_dir
        var disk_pt = _sample_disk_perpendicular(emit_dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        # Same derivation as distant, but pdf_dir is the CDF's solid-angle pdf
        # instead of an implicit delta (=1): flux = radiance × disk_area × n_lights / pdf_dir.
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, env_rgb.r, env_rgb.g, env_rgb.b, wavelengths) * (Float32(n_lights) * PI * radius * radius / pdf_dir)
        ro = disk_pt
        rd = emit_dir
        n_verts = 0
        # VCM Stage 2b MIS origin for a NON-FINITE (environment) light.
        # SmallVCM vertexcm.hxx GenerateLightSample, background branch:
        #     directPdfW   = pdf_dir * lightPickProb
        #     emissionPdfW = directPdfW / diskArea
        #     dVCM = directPdfW / emissionPdfW = diskArea
        #     dVC  = usedCosLight / emissionPdfW, usedCosLight = 1 (not finite)
        #     dVM  = dVC * mis_vc_weight_factor
        # lightPickProb = 1/n_lights cancels out of dVCM and appears in dVC
        # as the n_lights factor, exactly as in `flux` just above.
        #
        # These used to be left at 0 -- the scoped simplification the block
        # comment at the top of this function describes. Zero carries collapse
        # every MIS denominator to 1, so both light-side techniques took FULL
        # credit for illumination the camera side had already reported through
        # its env NEE. Measured on env-furnace (analytic answer 0.5): correct
        # at 0.5037 with both disabled, 0.5676 with t=1 splatting re-enabled
        # (+12.8%), 0.6701 with merging too (+20.5%) -- the whole of VCM's
        # +34% on every env-lit scene. Area lights were never affected (they
        # get real carries above), which is why only env-lit cells showed it.
        #
        # `is_finite_origin` deliberately stays False: that is what suppresses
        # the first-segment dist^2 factor below, correct here because the disk
        # is a sampling device, not a real emitter position.
        var disk_area = PI * radius * radius
        dvcm_carry = disk_area * Float32(n_lights)
        dvc_carry = disk_area * Float32(n_lights) / pdf_dir
        dvm_carry = dvc_carry * mis_vc_weight_factor
    else:
        # Point light: a real finite position (unlike distant/infinite), but
        # still no NEE-equivalent cache vertex — see this function's own
        # docstring for why (same reasoning as distant/infinite: direct
        # illumination comes from _bdpt_trace_camera_and_connect's own
        # per-vertex point-light NEE instead). Emits uniformly over the
        # sphere (isotropic point light); pdf_dir = 1/(4π), so
        # flux = intensity × 4π × n_lights (pdf_dir cancels).
        var pll = sd.pointLights[unsafe_offset=light_pick - n_area - n_distant - n_infinite]
        var u1p = pcg.next_float(); var u2p = pcg.next_float()
        var cos_p = Float32(1) - Float32(2) * u1p
        var sin_p = sqrt(max(Float32(0), Float32(1) - cos_p * cos_p))
        var phi_p = Float32(2) * PI * u2p
        var pdir_p = Vec3f(sin_p * cos(phi_p), sin_p * sin(phi_p), cos_p)
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, pll.intensity.r, pll.intensity.g, pll.intensity.b, wavelengths) * (Float32(4) * PI * Float32(n_lights))
        ro = pll.position
        rd = pdir_p
        n_verts = 0

    var cur_med_idx = default_emit_med
    var n_lbounces = 1  # counts all surface hits (1 = not-the-primary-ray, matches area-light convention — see _dielectric_bounce's bounce==0 special case)

    return VCMLightPathState_C(
        ro, rd, flux, dvcm_carry, dvc_carry, dvm_carry,
        Int8(1) if is_finite_origin else Int8(0),
        cur_med_idx, Int32(n_lbounces), Int32(n_verts), Int8(1),
        pcg.state, pcg.inc,
        wavelengths.lambda0, wavelengths.lambda1, wavelengths.lambda2, wavelengths.lambda3, wavelengths.pdf,
        Float32(1.0), Float32(1.0),   # current_dielectric_ior, previous_dielectric_ior (vacuum)
    )

def _bdpt_light_path_bounce[use_gpu: Bool](
    ref sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    inter: Intersection_C,
    lvc:      Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:   Int,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    mut ro: Point3f,
    mut rd: Vec3f,
    mut flux: SpectralSample,
    mut n_verts: Int,
    mut dvcm_carry: Float32,
    mut dvc_carry: Float32,
    mut dvm_carry: Float32,
    is_finite_origin: Bool,
    mut cur_med_idx: Int32,
    mut n_lbounces: Int,
    mut current_dielectric_ior: Float32,
    mut previous_dielectric_ior: Float32,
    wavelengths: SampledWavelengths,
    # Bump/normal-map footprint -- see _bdpt_camera_path_bounce's matching
    # params. The camera reference is deliberate on a LIGHT subpath: it is
    # pbrt's own choice for a vertex with no differentials, and it is the
    # only one that makes this vertex filter the surface the same way the
    # camera vertex it will be merged or connected with does.
    cam_pos: Vec3f,
    px_scale: Float32,
) -> Bool:
    """Task #163 stage 4: wavefront-staged variant of ONE bounce iteration of
    `_bdpt_trace_light_path`'s main loop (bdpt.mojo:1882-2383), split out so a
    GPU host loop can interleave a separate batched intersect dispatch
    (traverse_paths_gpu today, Vulkan RT via vulkaninterop later) between
    calls instead of tracing the whole subpath inside one kernel invocation.

    The material-dispatch BODY below (lines mirroring 1892-2380 of
    `_bdpt_trace_light_path`) is a byte-for-byte copy of that function's loop
    body -- see this module's VCM Stage 2b/2c comments there for the MIS
    derivation, not repeated here. The only changes from the original are
    mechanical, control-flow-shape ones:
      - the per-bounce intersect (`traverse_bvh2_core`/`test_spheres`) is
        REMOVED -- the caller supplies `inter` already computed via a
        separate batched dispatch.
      - every `break` that terminated the ORIGINAL function's outer bounce
        loop (7 sites: coat total-internal-reflection, coat walk exit
        failure, conductor invalid sample, 3x measured-BxDF failure, the
        unhandled-material-type catch-all) becomes `return False`.
      - the ORIGINAL function's 2 outer-loop `continue` sites (volume
        free-flight scatter, rough-coat reflect) become `return True`.
      - the coat recycling walk's OWN inner `for depth in
        range(MAX_COAT_DEPTH):` loop keeps its own real `break` statements
        (early-exit on RR kill, on finding a valid exit direction) --
        untouched, since those exit the INNER walk loop, not this function.
      - `n_verts`/loop-carried locals are `mut` PARAMETERS instead of
        function-local variables persisted implicitly across loop
        iterations -- the caller is a persistent per-light-path state
        struct (`VCMLightPathState_C`) that survives across separate kernel
        launches, one call to this function per bounce.
      - `lvc_path_len[lp_idx]` is NOT written here (unlike the original,
        which wrote it once after the loop) -- the caller writes
        `lvc_path_len[lp_idx] = Int32(n_verts)` unconditionally after every
        call, on both `return True` and `return False`, which is simpler
        and correct on every exit path without needing to track exactly
        which one fired.

    Returns True if the light path should continue to another bounce, False
    if it has terminated (hit nothing, exhausted a material's valid-sample
    conditions, or hit an unhandled material type)."""
        if n_verts >= _BDPT_MAX_VERTS:
            return False   # mirrors the original loop's top-of-iteration guard
        if inter.hit == Int8(0):
            return False   # nothing hit -- path escapes the scene
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
            var med = sd.mediums[unsafe_offset=Int(cur_med_idx)]
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            # density field. This call site used to be the homogeneous one
            # unconditionally, which rendered every "uniformgrid"/"nanovdb"/
            # "cloud" medium as uniform density-1 fog -- bunny-cloud came out a
            # featureless sphere with no bunny in it. See sample_free_flight.
            var ff = sample_free_flight(
                med, sd.grids, sd.nvdbGrids, Vec3f(ro.x, ro.y, ro.z), rd, t_hit, pcg)
            if ff.collided:
                # Chromatic collision weight; see the camera-side comment.
                flux *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                var sp = ro + rd*ff.t_free
                var v = _null_vertex()
                v.pos = sp
                # `flux`, NOT flux*albedo -- see the matching camera-side
                # comment in _bdpt_camera_path_bounce: a stored vertex's
                # beta/flux must EXCLUDE its own local response, which
                # _eval_vertex_spectral applies fresh at connect/merge time.
                v.beta = flux
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.pdf   # the density actually sampled from; under hero-wavelength MIS this is a lane MIXTURE, not sig_t's lone exponential
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                n_verts += 1
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                # Distant/infinite/point lights store NO lv0 (n_verts starts at
                # 0 for them -- this function's own docstring: they have no
                # NEE-equivalent cache vertex at all), so THIS volume scatter
                # can land at index 0 and is the ONLY bidirectional-connection
                # pathway those light types have into a medium -- deleting it
                # outright (an earlier version of this fix) broke them (an
                # env-lit slab went from 1.35x to 0.44x pbrt). An AREA-light
                # origin, by contrast, has the light-source vertex ITSELF at
                # index 0, and _bdpt_connect_to_cache restricts a camera
                # VOLUME vertex to connecting to THAT one only -- see its own
                # comment for why connecting to later same-subpath volume
                # vertices too over-counted an n-scatter path (n+1) times
                # (measured 2.7x on an area-lit slab). That restriction is
                # keyed on the light path's own origin, so storing every
                # volume vertex here unconditionally is correct for both.
                # Volume scatter is isotropic (no surface normal, no
                # cosThetaFix) -- out of MIS scope (like delta dielectric),
                # reset as if specular so a LATER diffuse/conductor/hair/
                # measured bounce still gets a well-defined carry.
                dvcm_carry = Float32(0)
                # Continuation: flux = prev × alb_s (same as stored vertex beta)
                flux *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ff.albedo).r, (ff.albedo).g, (ff.albedo).b, wavelengths)
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                return True   # volume free-flight scatter: no vertex stored this bounce, path continues
            else:
                flux *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # Same missing free-flight factor on dVCM as the camera path --
                # see _bdpt_camera_path_bounce's matching comment for the
                # derivation and the measured error growth with optical depth.
                var ff_exp_l = -log(max(ff.pdf, Float32(1e-30)))   # optical depth of the density actually sampled from (see FreeFlight.pdf)
                if ff_exp_l > Float32(60.0): ff_exp_l = Float32(60.0)
                dvcm_carry *= exp(ff_exp_l)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = ro + rd*t_hit

        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[unsafe_offset=mix_chosen]
            mat_idx = mix_chosen  # keep in sync with the resolved sub-material (hair needs the real index to re-fetch at connect time)
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

        if mat.type == MatKind.diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            # VCM Stage 2b: finish the per-bounce MIS correction (the
            # dist² portion was already applied above, shared across
            # branches) -- cos_fix is the incoming ray's cosine against
            # this vertex's own normal, matching SmallVCM's cosThetaFix.
            var cos_fix = abs(dot(-ray_dir, gn))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix)
            # Real image-texture reflectance -- see the matching comment in
            # _bdpt_trace_camera_and_connect's diffuse branch. Also feeds
            # the continuation flux multiply below, not just the stored
            # vertex, since both must agree on this bounce's actual albedo.
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            # Bump/normal maps -- see the camera-side diffuse branch.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_geo = gn
            gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn, gn, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn_geo)
            v.shading_normal = vec3f(gn)
            v.beta = flux
            v.alb = eff_alb
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            # diffusetransmission has a transmit lobe; without its own
            # LobeKind the vertex is re-evaluated as opaque Lambertian and
            # loses exactly half its energy. mat_idx is required: the
            # transmittance lives in Material_C.emission, not on the vertex.
            if mat.type == MatKind.diffuse_transmit:
                v.mat_kind = LobeKind.diffuse_transmit
                v.mat_idx = Int32(mat_idx)
            v.pdf_fwd = Float32(1); v.med_idx = cur_med_idx
            v.wo = vec3f(-ray_dir)  # VCM Stage 2b: needed for _connect's reverse-pdf eval
            v.wavelengths = wavelengths
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            n_verts += 1
            _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
            # Scatter
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            # diffusetransmission scatters to BOTH sides, with the lobe
            # picked by luminance -- the SAME split lobe_eval reports as the
            # density. Sampling one-sidedly while evaluating two-sidedly
            # leaves VCM's carries describing a path that was never built.
            # p_sel == 1 for every other material, so this is a no-op there
            # (and draws no extra random number).
            var p_sel = Float32(1.0)
            var bounce_n = gn
            var lobe_alb_dt = eff_alb
            if mat.type == MatKind.diffuse_transmit:
                var trans_dt = eff_alb if Int(mat.tex_idx) != -1 else mat.emission
                var pr_dt = eff_alb.luma()
                var pt_dt = trans_dt.luma()
                var tot_dt = max(pr_dt + pt_dt, Float32(1e-9))
                var take_refl = pcg.next_float() < pr_dt / tot_dt
                p_sel = max((pr_dt if take_refl else pt_dt) / tot_dt, Float32(1e-6))
                bounce_n = gn if take_refl else (gn * Float32(-1.0))
                lobe_alb_dt = eff_alb if take_refl else trans_dt
            rd = vec3f(_cosine_hemisphere_sample(bounce_n, u1, u2))
            ro = hit + rd*Float32(0.0002)
            flux *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (lobe_alb_dt).r, (lobe_alb_dt).g, (lobe_alb_dt).b, wavelengths) * (Float32(1.0) / p_sel)
            # VCM Stage 2b: recursive continuation for the NEXT bounce
            # (cosThetaOut/bsdfDirPdfW simplifies to PI exactly for
            # cosine-weighted diffuse sampling; bsdfRevPdfW reuses cos_fix,
            # the same incoming cosine just used above -- see the memory
            # file for the full derivation).
            var cos_theta_out = abs(dot(rd.to_simd(), bounce_n))
            # See the camera-side twin: p_sel on both densities, PI/p_sel for
            # cosThetaOut/bsdfDirPdfW.
            var bsdf_rev_pdf_w = p_sel * cos_fix / PI
            var bsdf_dir_pdf_w = p_sel * cos_theta_out / PI
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                dvcm_carry, dvc_carry, dvm_carry, PI / p_sel, bsdf_dir_pdf_w, bsdf_rev_pdf_w,
                mis_vc_weight_factor, mis_vm_weight_factor)

        elif mat.type == MatKind.coated_diffuse:
            # VCM light-side coateddiffuse (task #158) -- same sampling
            # geometry as the camera-side branch above (coat reflect vs
            # transmit-into-coat recycling walk), but WITHOUT any NEE: light
            # subpaths in this LVC architecture never do their own NEE (see
            # every other light-side material branch in this function --
            # only camera vertices call _bdpt_nee_contribute), so only the
            # walk's continuation direction + flux attenuation matter here.
            # Stored vertices use mat_kind=4, same unweighted scope as the
            # camera side.
            var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var cos_fix = abs(dot(-ray_dir, gn))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix)
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            # Bump/normal maps -- see the diffuse branch above.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_geo = gn
            gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn, gn, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))

            var ior = mat.emission.r
            var coat_alpha = max(mat.roughU, mat.roughV)
            var is_rough_coat = coat_alpha > Float32(0.001)
            var wo = -ray_dir
            # THE shared layered-BSDF walk (bxdf.mojo) -- see the camera-side
            # branch above. This light-side copy is gone too; only the LVC
            # vertex storage and the deliberate absence of eta^2 stay here.
            var cw = coat_walk_begin(gn, wo, eff_alb, ior, coat_alpha, pcg)
            coat_walk_enter(cw, pcg)

            if cw.event == COAT_ABSORB:
                return False   # coat total-internal-reflection-like case: terminate

            if cw.event == COAT_REFLECT:
                var refl = cw.wi
                rd = vec3f(refl)
                ro = hit + rd*Float32(0.0002)
                if is_rough_coat:
                    # cw.beta carries G2(wo,wi)/G1(wo); achromatic here.
                    flux *= cw.beta.r
                    var v = _null_vertex()
                    v.pos = hit
                    v.normal = vec3f(gn_geo)
                    v.shading_normal = vec3f(gn)
                    v.beta = flux
                    v.alb = eff_alb
                    # coated_REFLECT -- the light side's twin of the camera
                    # branch above; this half reaches the photon cache, so a
                    # wrong evaluator here misweights every merge that
                    # gathers it.
                    v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.coated_reflect
                    v.mat_idx = Int32(mat_idx)   # the coat evaluator reads ior from it
                    v.pdf_bwd = coat_alpha       # smooth-vs-rough, as ggx stores alpha
                    v.wo = vec3f(wo)
                    # Real density and real carries -- the camera side's twin
                    # of this vertex carries the full story. This half is the
                    # one that reaches the photon cache, so fabricated carries
                    # here corrupt the weight of every MERGE that gathers this
                    # vertex, not just this subpath's own connections.
                    v.pdf_fwd = cw.pdf
                    v.med_idx = cur_med_idx
                    v.wavelengths = wavelengths
                    v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                    n_verts += 1
                    _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                    var cos_out_lr = abs(dot(refl, gn))
                    var (_pf_lr, pdf_rev_lr) = _bdpt_vertex_pdfs(v, vec3f(refl), sd)
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                        dvcm_carry, dvc_carry, dvm_carry,
                        cos_out_lr / max(cw.pdf, Float32(1e-9)), cw.pdf, pdf_rev_lr,
                        mis_vc_weight_factor, mis_vm_weight_factor)
                else:
                    # Genuinely delta (smooth mirror coat): dVCM resets.
                    dvcm_carry = Float32(0)
                return True   # coat-reflect (rough): vertex already stored above, path continues

            # Entry attenuation already applied by coat_walk_enter. The light
            # subpath does no NEE of its own, so this loop is just the walk.
            while cw.event == COAT_WALKING:
                if not coat_walk_at_base(cw, pcg):
                    break
                coat_walk_scatter(cw, pcg)

            var exited = cw.event == COAT_EXIT
            var exit_dir = cw.wi
            if not exited:
                return False   # coat recycling walk failed to find an exit direction

            rd = vec3f(exit_dir)
            ro = hit + rd*Float32(0.0002)
            # NO 1/eta^2 here, unlike the camera-side exit -- deliberately.
            # The camera-side factor accounts for radiance compression
            # leaving a denser medium under RADIANCE transport; a light
            # subpath carries IMPORTANCE, whose non-symmetric scattering
            # correction runs the other way (see Veach ch. 5 / pbrt's
            # TransportMode::Importance BTDF branch) and is entangled with
            # this codebase's own documented, already-accepted gap for
            # coateddiffuse BDPT vertices: they're stored as mat_kind=4,
            # which _bdpt_vertex_mis_scoped excludes from real per-vertex
            # MIS, so connect/merge reach them through _eval_vertex's
            # generic unweighted fallback rather than a proper eta-aware
            # BDPT connection weight (same shape as the dielectric/volume
            # gap noted elsewhere in this file). Guessing a sign here
            # without that connection-weight context fixed first risks
            # trading one silent bias for another; left alone, not ignored.
            # The stored PHOTON carries the flux arriving BEFORE the coat, because a
            # merge or connection at this vertex supplies the coat transport itself
            # through coat_eval_smooth. Storing the post-walk flux applies the coat
            # TWICE -- the light-side twin of the camera-side double application fixed
            # in ba3ea22b. Every factor in it is < 1, so it reads as too DARK.
            # The continuing photon still carries the post-walk flux.
            # Light-side twin of 6ec79292: a smooth coat's exit density is known,
            # so this photon gets REAL carries and its path keeps them. Zero carries
            # here made every merge-with and t=1 weight at a coated vertex too large.
            var cos_x_l = abs(dot(exit_dir, gn))
            var smooth_coat_l = coat_alpha <= Float32(0.001)
            var pdf_x_l = bxdf_pdf_coated_exit(cos_x_l, ior)
            var flux_pre_coat = flux
            flux *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (cw.beta).r, (cw.beta).g, (cw.beta).b, wavelengths)
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn_geo)
            v.shading_normal = vec3f(gn)
            v.beta = flux_pre_coat
            v.alb = eff_alb
            v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.coated_walk
            v.mat_idx = Int32(mat_idx)   # the coat evaluator reads ior from it
            v.pdf_bwd = coat_alpha       # smooth-vs-rough, as ggx stores alpha
            v.wo = vec3f(wo)
            v.pdf_fwd = pdf_x_l
            v.med_idx = cur_med_idx
            v.wavelengths = wavelengths
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            n_verts += 1
            _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                dvcm_carry, dvc_carry, dvm_carry,
                cos_x_l / max(pdf_x_l, Float32(1e-9)), pdf_x_l,
                bxdf_pdf_coated_exit(abs(dot(wo, gn)), ior),
                mis_vc_weight_factor, mis_vm_weight_factor)

        elif mat.type == MatKind.conductor or mat.type == MatKind.coated_conductor:
            var gn_c = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            # Bump/normal maps -- see the diffuse branch.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_c_geo = gn_c
            gn_c = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn_c, gn_c, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=Vec3f(frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=Vec3f(frm_c.y.x, frm_c.y.y, frm_c.y.z),
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
                return False   # conductor/coated_conductor sample invalid
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
                v.normal = vec3f(gn_c_geo)
                v.shading_normal = vec3f(gn_c)
                v.beta = flux
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.ggx
                v.pdf_bwd = alpha_c
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                n_verts += 1
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
            flux *= spec_refl_unbounded(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (bs_c.f).r, (bs_c.f).g, (bs_c.f).b, wavelengths)
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
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                        dvcm_carry, dvc_carry, dvm_carry, inv_pdf_c, bsdf_dir_pdf_w_c, bsdf_rev_pdf_w_c,
                        mis_vc_weight_factor, mis_vm_weight_factor)
                else:
                    dvcm_carry = Float32(0)
                    dvc_carry = Float32(0)
                    dvm_carry = Float32(0)

        elif mat.type == MatKind.hair:
            var curve_idx_h = Int(inter.primId.id1)
            var wo_h = (-rd).to_simd()
            var hc = _hair_precompute(mat, sd.curves, curve_idx_h, inter.v, inter.u, wo_h)
            # VCM: hair DOES have a real standalone pdf -- see the camera-
            # side hair branch's matching comment and _bdpt_vertex_pdfs'
            # mat_kind=2 branch for the full derivation.
            var cos_fix_h = abs(dot(-ray_dir, hc.geo_normal))
            if cos_fix_h > Float32(1e-6):
                dvc_carry /= cos_fix_h
                dvm_carry /= cos_fix_h
            var v_h = _null_vertex()
            v_h.pos = hit
            v_h.normal = vec3f(hc.geo_normal)
            v_h.shading_normal = vec3f(hc.geo_normal)
            v_h.beta = flux
            v_h.alb = mat.albedo
            v_h.is_surface = Int32(1); v_h.is_delta = Int32(0); v_h.mat_kind = LobeKind.hair
            v_h.wo = vec3f(wo_h)
            v_h.mat_idx = Int32(mat_idx)
            v_h.hair_curve_idx = Int32(curve_idx_h)
            v_h.hair_h = inter.u
            v_h.hair_v = inter.v
            v_h.pdf_fwd = Float32(1)
            v_h.med_idx = cur_med_idx
            v_h.wavelengths = wavelengths
            v_h.dVCM = dvcm_carry; v_h.dVC = dvc_carry; v_h.dVM = dvm_carry
            n_verts += 1
            _bdpt_store_lvc_vertex(v_h, lvc, lp_idx, n_verts - 1)
            var (wi_hs, f_hs, pdf_hs, cos_ti_hs2) = _hair_sample_dir(hc, pcg)
            flux *= spec_refl_unbounded(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (f_hs / pdf_hs).r, (f_hs / pdf_hs).g, (f_hs / pdf_hs).b, wavelengths)
            rd = vec3f(wi_hs)
            var hsign = Float32(1) if dot(wi_hs, hc.geo_normal) >= Float32(0) else Float32(-1)
            ro = hit + vec3f(hc.geo_normal) * curve_offset_eps(hc.radius) * hsign
            var cos_theta_out_h = abs(dot(wi_hs, hc.geo_normal))
            # VCM: real non-specular recursive update -- see the camera-side
            # hair branch's matching comment for the full derivation (same
            # formula, light-path variant: no last_bsdf_pdf bookkeeping here
            # since light paths don't do infinite-light-miss MIS).
            var bsdf_dir_pdf_w_h = pdf_hs * cos_ti_hs2
            if bsdf_dir_pdf_w_h > Float32(1e-8):
                var hc_rev_h = _hair_precompute(mat, sd.curves, curve_idx_h, inter.v, inter.u, wi_hs)
                var (cos_ti_rev_h, _, pdf_oc_rev_h) = _hair_eval_lobes(
                    wo_h, hc_rev_h.tangent, hc_rev_h.b_perp, hc_rev_h.n_perp, hc_rev_h.phi_o,
                    hc_rev_h.dphi0, hc_rev_h.dphi1, hc_rev_h.dphi2,
                    hc_rev_h.cos_tp0_o, hc_rev_h.sin_tp0_o, hc_rev_h.cos_tp1_o, hc_rev_h.sin_tp1_o, hc_rev_h.cos_tp2_o, hc_rev_h.sin_tp2_o,
                    hc_rev_h.cos_theta_o, hc_rev_h.sin_theta_o, hc_rev_h.inv_vm0, hc_rev_h.inv_vm1, hc_rev_h.inv_vm2, hc_rev_h.mp_c0, hc_rev_h.mp_c1, hc_rev_h.mp_c2, hc_rev_h.s,
                    hc_rev_h.A0, hc_rev_h.A1, hc_rev_h.A2, hc_rev_h.A3, hc_rev_h.lum0, hc_rev_h.lum1, hc_rev_h.lum2, hc_rev_h.lum3, hc_rev_h.total_lum,
                )
                var bsdf_rev_pdf_w_h = cos_ti_rev_h * pdf_oc_rev_h
                var inv_pdf_h = cos_theta_out_h / bsdf_dir_pdf_w_h
                (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                    dvcm_carry, dvc_carry, dvm_carry, inv_pdf_h, bsdf_dir_pdf_w_h, bsdf_rev_pdf_w_h,
                    mis_vc_weight_factor, mis_vm_weight_factor)
            else:
                dvcm_carry = Float32(0)
                dvc_carry = Float32(0)
                dvm_carry = Float32(0)

        elif mat.type == MatKind.measured:
            # Mirrors the camera-side measured branch above, minus the NEE
            # loops (light subpaths don't do NEE against other lights) --
            # see that branch's docstring for the shared-interface rationale.
            var gn_m = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn_m, ray_dir) > Float32(0): gn_m = gn_m * Float32(-1)
            # Bump/normal maps -- see the diffuse branch.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_m_geo = gn_m
            gn_m = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn_m, gn_m, ray_dir,
                _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            if mat.measured_idx < Int32(0):
                return False   # measured: no tabulated BRDF for this material
            var wo_m = (-rd).to_simd()
            var frm_m = Frame.from_z(Vec3f(gn_m[0], gn_m[1], gn_m[2]))
            var tangent_m = Vec3f(frm_m.x.x, frm_m.x.y, frm_m.x.z)
            var bitangent_m = Vec3f(frm_m.y.x, frm_m.y.y, frm_m.y.z)
            var mb = sd.measuredBrdfs[unsafe_offset=Int(mat.measured_idx)]
            var wo_l_m = Vec3f(dot(wo_m, tangent_m), dot(wo_m, bitangent_m), dot(wo_m, gn_m))
            var uml1 = pcg.next_float(); var uml2 = pcg.next_float()
            var (wi_l_m, f_m, pdf_m, valid_m) = bxdf_sample_measured(mb, wo_l_m, uml1, uml2, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
            if not valid_m or pdf_m <= Float32(0):
                return False   # measured: invalid sample
            # VCM Stage 2b: measured BxDF DOES have a real standalone pdf
            # (bxdf_pdf_measured, unlike conductor/hair) -- in MIS scope
            # this pass, same real treatment as diffuse (see that branch's
            # comments + project_vcm_stage2_mis_derivation memory).
            var cos_fix_m = abs(dot(-ray_dir, gn_m))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix_m)
            var v_m = _null_vertex()
            v_m.pos = hit
            v_m.normal = vec3f(gn_m_geo)
            v_m.shading_normal = vec3f(gn_m)
            v_m.beta = flux
            v_m.alb = mat.albedo
            v_m.is_surface = Int32(1); v_m.is_delta = Int32(0); v_m.mat_kind = LobeKind.measured
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
                return False   # measured: sampled direction below the surface
            # f_m is already spectral -- no RGB round trip.
            flux *= f_m * (cos_wi_m / pdf_m)
            rd = vec3f(wi_m)
            ro = hit + rd*Float32(0.0002)
            # VCM Stage 2b: recursive continuation, real forward/reverse pdf
            # from bxdf_pdf_measured (reverse = same call with wo/wi swapped
            # -- ASSUMED convention, not independently verified against
            # measured_bxdf_eval.mojo's exact semantics; flag if results look
            # wrong on sportscar/measured-material scenes).
            var pdf_rev_m = bxdf_pdf_measured(mb, wi_l_m, wo_l_m)
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries(
                dvcm_carry, dvc_carry, dvm_carry, (cos_wi_m / pdf_m), pdf_m, pdf_rev_m,
                mis_vc_weight_factor, mis_vm_weight_factor)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # ── Subsurface boundary: the light subpath's half of the hop ───
            # Mirror of the camera side: the light enters at x_i, the exit x_o
            # is drawn by the SAME sampler (the MIS weights require identical
            # densities on both sides; the hop is symmetric), x_o is stored as
            # a connectible light vertex with the exit lobe, and the subpath
            # continues from it. The entry is never stored.
            var did_bssrdf_hop = False
            if mat.sss_boundary != Int8(0) and Int(cur_med_idx) < 0 and has_med:
                var med_in = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if med_in >= Int32(0) and Int(med_in) < Int(sd.mediumCount):
                    var gn_e = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                    if dot(gn_e, ray_dir) > Float32(0): gn_e = gn_e * Float32(-1)
                    var cos_e = abs(dot(-ray_dir, gn_e))
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                        dvcm_carry, dvc_carry, dvm_carry, cos_e)
                    var eta_e = mat.albedo.r
                    var ex = _bdpt_sample_bssrdf_exit(sd, Int(med_in), hit, gn_e, cos_e, eta_e, pcg)
                    if not ex.ok:
                        return False
                    flux *= spec_refl_unbounded(
                        sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                        sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                        ex.weight.r, ex.weight.g, ex.weight.b, wavelengths)
                    var (h0, h1, h2) = bssrdf_hop_carries(
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_e * INV_PI, mis_vc_weight_factor)
                    dvcm_carry = h0
                    dvc_carry = h1
                    dvm_carry = h2
                    var n_o = ex.n_o
                    var v = _null_vertex()
                    v.pos = ex.x_o
                    v.normal = vec3f(n_o)
                    v.shading_normal = vec3f(n_o)
                    v.beta = flux
                    v.alb = RGB(Float32(1))
                    v.is_surface = Int32(1); v.is_delta = Int32(0)
                    v.mat_kind = LobeKind.bssrdf
                    v.pdf_fwd = ex.p_area
                    v.pdf_bwd = eta_e
                    v.wo = vec3f(n_o)
                    v.med_idx = cur_med_idx
                    v.wavelengths = wavelengths
                    v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                    n_verts += 1
                    _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                    var ux1 = pcg.next_float(); var ux2 = pcg.next_float()
                    rd = vec3f(_cosine_hemisphere_sample(n_o, ux1, ux2))
                    ro = ex.x_o + rd * Float32(0.0002)
                    var cos_out_x = abs(dot(rd.to_simd(), n_o))
                    flux *= SpectralSample(bssrdf_exit_ft(cos_out_x, eta_e))
                    var (c0, c1, c2) = bssrdf_exit_scatter_carries(
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_out_x,
                        mis_vc_weight_factor, mis_vm_weight_factor)
                    dvcm_carry = c0
                    dvc_carry = c1
                    dvm_carry = c2
                    n_lbounces += 1
                    did_bssrdf_hop = True
            if not did_bssrdf_hop:
                var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                # Bump/normal maps. barcelona's water is a `dielectric` with a
                # "texture displacement", so this is THE branch the corpus
                # cares about most -- and VCM refracted through a perfectly
                # flat sheet while the path tracer refracted through a
                # rippled one. `orient_to` is the RAW winding normal, NOT a
                # face-forwarded one: _dielectric_bounce decides entering vs
                # exiting from dot(ray_dir, n) < 0, and flipping the
                # perturbed normal toward the ray would make that test
                # tautological and resurrect the 1/eta^4 transmission loss
                # (same rule as shading.mojo's dielectric site).
                gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                    gn, gn, ray_dir,
                    _camera_approx_footprint(hit, cam_pos, px_scale),
                    sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
                var (new_dir, new_org, _, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                    ray_dir, hit.to_simd(), gn, mat.albedo.r, n_lbounces == 0 and Int(cur_med_idx) < 0, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric, radiance_mode=False)
                current_dielectric_ior = new_cur_ior
                previous_dielectric_ior = new_prev_ior
                n_lbounces += 1
                # Light path (TransportMode::Importance): do NOT apply the
                # radiance_scale non-symmetric-scattering correction — it's only
                # for camera/Radiance-mode paths, see _dielectric_bounce's
                # docstring.
                if has_med:
                    var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
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
                var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)
            # VCM Stage 2b: pure medium-boundary pass-through, no direction
            # change/BSDF event -- carry state passes through as already
            # dist²-corrected above (the ray genuinely traveled t_hit), no
            # cosFix/reset applied since there's no real "bounce" here.

        else:
            return False   # unhandled material type


        return True   # bounce processed normally, path continues

def _bdpt_trace_light_path[use_gpu: Bool](
    ref sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    default_emit_med: Int32,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    lvc:      Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:   Int,
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    pass_wl: SampledWavelengths,
    # Bump/normal-map footprint reference -- forwarded straight to
    # _bdpt_light_path_bounce, see its own params. Deliberately NOT defaulted:
    # a px_scale of 0 means "no footprint", which _apply_bump_map answers with
    # a FIXED 0.0005 UV step -- the exact failure that made barcelona's water
    # worse the first time bump mapping was tried. A missed call site must be
    # a compile error, not a silently unbumped light subpath.
    cam_pos: Vec3f,
    px_scale: Float32,
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

    # SHARED WITH THE WAVEFRONT GPU DRIVER. This body used to be a
    # byte-for-byte copy of _bdpt_light_path_init + _bdpt_light_path_bounce
    # -- the two halves task #163 split out for wavefront staging -- so
    # every material/MIS/estimator change had to be made twice and silently
    # drifted whenever it wasn't. That drift is exactly why --vcm-wavefront
    # still lacks the t=1 splat and the Vulkan RT coverage fixes. Both
    # designs now run the SAME step: this one loops over it inline, the
    # wavefront driver launches it once per depth level with the loop-carried
    # state parked in a VCMLightPathState_C between launches.
    var st = _bdpt_light_path_init[use_gpu](
        sd, pcg, default_emit_med, lp_idx, lvc, lvc_path_len, mis_vc_weight_factor, pass_wl)
    if st.active == Int8(0):
        return
    var ro = st.ro
    var rd = st.rd
    var flux = st.flux
    var n_verts = Int(st.n_verts)
    var dvcm_carry = st.dvcm
    var dvc_carry = st.dvc
    var dvm_carry = st.dvm
    var is_finite_origin = st.is_finite_origin == Int8(1)
    var cur_med_idx = st.cur_med_idx
    var n_lbounces = Int(st.n_lbounces)
    var current_dielectric_ior = st.current_dielectric_ior
    var previous_dielectric_ior = st.previous_dielectric_ior
    var wavelengths = SampledWavelengths(st.wl0, st.wl1, st.wl2, st.wl3, st.wl_pdf)

    for _ in range(_BDPT_MAX_DEPTH):
        # The same intersect step _bdpt_light_path_intersect_gpu performs --
        # kept here rather than inside the bounce because this one step is
        # the eventual Vulkan RT swap point on the wavefront side.
        var ray = Ray_C(ro, rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        # A miss is handled inside the step (returns False), so the old
        # top-of-loop `if scratch[0].hit == 0: break` is no longer needed.
        if not _bdpt_light_path_bounce[use_gpu](
            sd, pcg, has_med, scratch[unsafe_offset=0], lvc, lp_idx,
            mis_vc_weight_factor, mis_vm_weight_factor,
            ro, rd, flux, n_verts, dvcm_carry, dvc_carry, dvm_carry,
            is_finite_origin, cur_med_idx, n_lbounces,
            current_dielectric_ior, previous_dielectric_ior, wavelengths,
            cam_pos, px_scale):
            break

    lvc_path_len[unsafe_offset=lp_idx] = Int32(n_verts)
    return

# ── BSDF/phase evaluation at a vertex ────────────────────────────────────────

# The RGB siblings of these two (_eval_vertex / _eval_conductor_ggx) are gone:
# BDPT/VCM transport is spectral, so a connection multiplies two spectral
# betas and there is no RGB product left for an RGB evaluator to feed.
#
# These take SpectralHandle's fields DECOMPOSED into individual pointer/int
# params, NOT a single by-value SpectralHandle param -- passing that 6-field
# struct by value across a real Mojo function-call boundary was suspected of
# a miscompilation (modular/modular#6759, later retracted as unreproducible;
# see spectrum.mojo's comment above rgb_to_spectral_sample), kept decomposed
# defensively. Hair (mat_kind=2) is NOT covered here (same
# deliberate exclusion as bxdf.mojo's spectral siblings) -- callers must
# check v.mat_kind != 2 before using these; _connect below does exactly
# that by falling back to the plain RGB _eval_vertex/_eval_conductor_ggx for
# any connection touching a hair vertex.
@fieldwise_init
struct BssrdfExitSample(TrivialRegisterPassable):
    """Result of sampling where a subsurface hop leaves the surface."""
    var ok: Bool
    var x_o: Point3f
    var n_o: Vec3f        # outward geometric normal at the exit
    var weight: RGB       # R_d(r) * Ft(entry) / p_A -- the exit lobe's Ft is NOT in here
    var p_area: Float32   # p_A(x_o | x_i), area measure; also the reverse density


def _bdpt_sample_bssrdf_exit(
    ref sd: SceneDescriptor2_C,
    med_idx: Int,
    hit: Point3f,
    n_in: Vec3f,          # entry normal, facing the side the path arrived from
    cos_in: Float32,
    eta: Float32,
    mut pcg: PCG32,
) -> BssrdfExitSample:
    """ONE exit-point sampler for both VCM subpaths. The MIS weights assume the
    camera and light sides draw the hop from the same density (the hop is
    symmetric, bssrdf.mojo), so this must never be duplicated per side.

    A channel is chosen uniformly and its exponential radius sampled; the pdf
    is the mixture over channels. The tangent-disk sample is projected onto
    the surface by a probe ray along the entry normal, whose |n_in . n_o|
    Jacobian is part of p_A. A missed probe or an exit beyond the profile's
    reach is a failed sample: the caller terminates the path (it must not fall
    back to a different strategy, which would bias the estimate).

    `sd` is taken by reference, like every other descriptor parameter here:
    by value, one call deeper than the light loop's own traversal, the probe
    traversal faulted with CUDA_ERROR_ILLEGAL_ADDRESS in
    _bdpt_emit_light_paths_gpu."""
    var med = sd.mediums[unsafe_offset=med_idx]
    var fail = BssrdfExitSample(False, hit, n_in, RGB(Float32(0)), Float32(0))
    var ft_in = bssrdf_exit_ft(cos_in, eta)
    if ft_in <= Float32(0.0):
        return fail
    var r_max = dipole_max_radius(med.sigma_s, med.sigma_a, med.g)
    var ch = Int(pcg.next_float() * Float32(3.0))
    if ch > 2: ch = 2
    var sigma_tr = dipole_mis_sigma_tr(med.sigma_s, med.sigma_a, med.g, ch)
    if sigma_tr <= Float32(0.0):
        return fail
    var r = dipole_sample_radius(sigma_tr, pcg.next_float())
    var phi = Float32(2.0) * PI * pcg.next_float()
    if r >= r_max:
        return fail
    var frm = Frame.from_z(n_in)
    var (off, seg_len) = bssrdf_probe_offset(
        r, phi, r_max, Vec3f(frm.x.x, frm.x.y, frm.x.z), Vec3f(frm.y.x, frm.y.y, frm.y.z), n_in)
    var probe_org = hit + off
    var probe_dir = n_in * Float32(-1.0)
    # Private probe slot: the caller's scratch still holds the intersection
    # the enclosing path loop is shading.
    var _probe_slot = InlineArray[Intersection_C, 1](fill=Intersection_C(
        PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var probe_scratch = _probe_slot.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    probe_scratch[unsafe_offset=0].hit = Int8(0)
    var probe_ray = Ray_C(probe_org, probe_dir)
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe_ray, seg_len, probe_scratch,
                       sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    test_spheres(sd.spheres, Int(sd.sphereCount), probe_ray, probe_scratch)
    if probe_scratch[unsafe_offset=0].hit == Int8(0):
        return fail
    var pi = probe_scratch[unsafe_offset=0]
    # The exit must be on a subsurface boundary too, or the profile does not
    # describe what happens there.
    var pmat = sd.materials[unsafe_offset=Int(pi.primId.materialIndex)]
    if pmat.sss_boundary == Int8(0):
        return fail
    var x_o = probe_org + probe_dir * pi.tHit
    var n_o: Vec3f
    if pi.primId.type == Int8(4):
        n_o = sphere_outward_normal(x_o, sd.spheres[unsafe_offset=Int(pi.primId.id1)].center)
    else:
        n_o = _geom_normal(pi, sd.meshes, sd.instances, sd.spheres, x_o.to_simd())
    if dot(n_o, n_in) < Float32(0.0):
        n_o = n_o * Float32(-1.0)
    var d = x_o - hit
    var r_act = sqrt(dot(d, d))
    if r_act > r_max:
        return fail
    var p_area = bssrdf_exit_pdf_area(med.sigma_s, med.sigma_a, med.g, r_act, dot(n_o, n_in))
    if p_area <= Float32(1e-12):
        return fail
    var rd = dipole_rd(med.sigma_s, med.sigma_a, med.g, eta, r_act)
    var k = ft_in / p_area
    return BssrdfExitSample(True, x_o, n_o, RGB(rd.r * k, rd.g * k, rd.b * k), p_area)



@always_inline
def _bdpt_n_lights(ref sd: SceneDescriptor2_C) -> Float32:
    """The light-pick denominator the light path used, needed on the camera
    side because its NEE loops every light with no pick: the MIS densities
    must describe the same experiment on both subpaths."""
    return Float32(Int(sd.areaLightCount) + Int(sd.distantLightCount)
                   + Int(sd.infiniteLightCount) + Int(sd.pointLightCount))

@always_inline
def _vertex_ctx(v: BDPTVertex) -> LobeCtx:
    """A stored VCM vertex, as the shared BxDF interface sees it."""
    return LobeCtx(v.mat_kind, v.is_surface == Int32(1), v.is_delta != Int32(0),
                   v.shading_normal.to_simd(), v.wo.to_simd(), v.alb, v.mat_idx,
                   v.pdf_bwd, v.pdf_fwd, v.hair_curve_idx, v.hair_h, v.hair_v, False)


@always_inline
def _lobe_eval[want_pdfs: Bool = True](
    v:   BDPTVertex,
    dir_to_other:  Vec3f,
    ref sd:  SceneDescriptor2_C,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
) -> LobeEval:
    """VCM's view of the shared lobe evaluator (bxdf.mojo)."""
    return lobe_eval[want_pdfs](_vertex_ctx(v), dir_to_other,
        LobeTables(sd.materials, sd.curves, sd.measuredBrdfs),
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, wavelengths)

@always_inline
def _eval_vertex_spectral(
    v:   BDPTVertex,
    dir_to_other:  Vec3f,
    ref sd:  SceneDescriptor2_C,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Throughput at `v` toward `dir_to_other`: the BSDF (or phase function)
    times this lobe's OWN cosine. A thin view onto _lobe_eval -- see
    LobeEval for why the dispatch it used to duplicate now lives in one
    place."""
    return _lobe_eval[want_pdfs=False](
        v, dir_to_other, sd, spectral_coeffs, spectral_res, spectral_cie_x,
        spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths).f_cos

@always_inline
def _bdpt_vertex_pdfs(
    v: BDPTVertex, dir_to_other: Vec3f, ref sd: SceneDescriptor2_C,
) -> Tuple[Float32, Float32]:
    """Forward/reverse solid-angle densities at `v` toward `dir_to_other`.

    A thin view onto _lobe_eval, which is now the single dispatch over
    LobeKind. This was one of three hand-maintained copies of that dispatch
    -- see LobeEval. Wavelengths come off the vertex because the density half
    never uses them; only the throughput half does, and that is what this
    call discards."""
    var le = _lobe_eval[want_pdfs=True](
        v, dir_to_other, sd, sd.spectral.coeffs, sd.spectral.res,
        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z,
        sd.spectral.d65, v.wavelengths)
    return (le.pdf_fwd, le.pdf_rev)

@always_inline
def _lobe_scoped(v: BDPTVertex) -> Bool:
    """VCM's view of the shared scope test (bxdf.mojo)."""
    return lobe_scoped(_vertex_ctx(v))

@always_inline
def _bdpt_vertex_mis_scoped(v: BDPTVertex) -> Bool:
    """Alias kept for call sites; see _lobe_scoped."""
    return _lobe_scoped(v)


@always_inline
def _bdpt_vertex_mis_scoped_kinds(v: BDPTVertex) -> Bool:
    """True for vertex kinds _bdpt_vertex_pdfs has a real pdf for: diffuse
    (mat_kind=0, real surface -- excludes volume vertices, which default
    to mat_kind=0 too, see _connect's matching comment), rough
    conductor/coated_conductor (mat_kind=1, always non-delta by
    construction -- delta conductor bounces are never stored as
    connectible vertices at all, see _bdpt_trace_camera_and_connect's
    conductor branch), hair (mat_kind=2, Marschner 3-lobe -- also always
    non-delta, no specular lobe exists in this model), and measured
    (mat_kind=3). Dielectric/thin_dielectric are NOT in this list and
    never will be without first adding rough-dielectric support -- they're
    genuinely delta/specular in this codebase (true reflect-or-refract,
    not an approximation), so they're never even stored as LVC vertices
    at all (see _bdpt_trace_camera_and_connect's/_bdpt_trace_light_path's
    dielectric branches), making this function unreachable for them by
    construction, not merely False."""
    if v.is_surface != Int32(1):
        return False
    return (v.mat_kind == LobeKind.lambertian or v.mat_kind == LobeKind.ggx or v.mat_kind == LobeKind.hair
            or v.mat_kind == LobeKind.measured or v.mat_kind == LobeKind.bssrdf
            or v.mat_kind == LobeKind.diffuse_transmit)

@always_inline
def _bdpt_connect_pair_weighted(cv: BDPTVertex, lv: BDPTVertex) -> Bool:
    """True when _connect applies a real per-pair MIS weight to (cv, lv).

    MUST stay identical to the condition guarding _connect's own dVCM/dVC
    weight block -- the caller uses this to decide which connections may be
    summed freely (weighted ones) and which must be limited to one per camera
    vertex (unweighted ones), so a mismatch here silently reintroduces the
    over-count this predicate exists to prevent."""
    return _bdpt_vertex_mis_scoped(cv) and (lv.is_light == Int32(1) or _bdpt_vertex_mis_scoped(lv))

# ── Connect one camera vertex to one light vertex ─────────────────────────────

def _connect(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    ref sd: SceneDescriptor2_C,
    has_med: Bool,
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    mis_vm_weight_factor: Float32,
) -> SpectralSample:
    """Evaluate the contribution of connecting cv to lv via a shadow ray.
    Each connection is already a complete, self-normalized estimator of its
    own depth-strategy's contribution (see the module's LVC-BPT docstring);
    the caller sums over every vertex of the paired light path with no
    further scaling.

    VCM Stage 2b/2d/153/hair (2026-07-10/11): real per-vertex MIS weighting
    (Georgiev et al. 2012 / SmallVCM, see project_vcm_stage2_mis_derivation
    memory) is applied when both endpoints have a genuine standalone pdf --
    diffuse (mat_kind=0), light-source vertices (is_light=1, whose
    cosine-weighted emission profile is mathematically the same shape as
    diffuse), rough conductor/coated_conductor (mat_kind=1, GGX-VNDF pdf),
    hair (mat_kind=2, Marschner 3-lobe pdf via _hair_eval_lobes -- the same
    pdf machinery _nee_weight_hair already trusts for NEE MIS), and
    measured (mat_kind=3, tabulated-BRDF pdf via a reconstructed local
    frame). Only volume (isotropic phase, no surface normal) falls through
    to `weight=1`, today's plain unweighted behavior; dielectric/
    thin_dielectric are genuinely delta/specular and never even reach here
    at all (never stored as LVC vertices, see this file's opening VCM
    comment) -- both deliberately scoped boundaries, not silent
    omissions."""
    # THE connection estimator lives in _connect_unweighted; this is that
    # times visibility. They used to be two byte-for-byte copies (one for the
    # deferred Vulkan-RT shadow-ray path, which resolves visibility later),
    # and a copy is how a fix lands in one and not the other -- see
    # feedback_unify_while_fixing. Evaluating the BSDFs first also skips the
    # shadow ray entirely for a connection that is zero anyway.
    var (contrib, valid) = _connect_unweighted(cv, lv, sd, mis_vm_weight_factor)
    if not valid or contrib.is_black():
        return SpectralSample(Float32(0))
    # Medium for the shadow segment: the camera vertex's (both endpoints agree
    # in a well-defined scene).
    var Tr = _visible_transmittance(cv.pos, lv.pos, cv.med_idx, sd, scratch, cv.wavelengths)
    if Tr.is_black():
        return SpectralSample(Float32(0))
    return contrib * Tr

def _connect_unweighted(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    ref sd: SceneDescriptor2_C,
    mis_vm_weight_factor: Float32,
) -> Tuple[SpectralSample, Bool]:
    """Task #163 stage 5: byte-for-byte copy of _connect's math (see that
    function's own docstring for the MIS derivation, not repeated here),
    with the visibility test (_visible_transmittance) REMOVED -- returns
    (contrib_unweighted, valid) instead of the final Tr-weighted
    contribution. `valid=False` means _connect's own early-exit conditions
    (delta endpoint, coincident points, light facing away) already prove
    the contribution is exactly zero regardless of visibility -- the
    caller should NOT bother queuing a shadow ray for these. `valid=True`
    means the caller must still resolve visibility (Tr) and multiply it
    into contrib_unweighted -- this function intentionally does not do
    that itself, so its result can be used to fill a batched Vulkan RT
    shadow-ray queue instead of resolving inline per-thread. Used ONLY by
    the wavefront-staged GPU path's deferred-connect kernels
    (_bdpt_connect_diffuse_deferred_gpu et al.) when use_vk=True -- _connect
    itself is UNCHANGED and still drives the CPU renderer, the
    non-wavefront GPU renderer, and the wavefront GPU renderer's own
    non-deferred (use_vk=False) path.

    VCM Stage 2b/2d/153/hair (2026-07-10/11): real per-vertex MIS weighting
    (Georgiev et al. 2012 / SmallVCM, see project_vcm_stage2_mis_derivation
    memory) is applied when both endpoints have a genuine standalone pdf --
    diffuse (mat_kind=0), light-source vertices (is_light=1, whose
    cosine-weighted emission profile is mathematically the same shape as
    diffuse), rough conductor/coated_conductor (mat_kind=1, GGX-VNDF pdf),
    hair (mat_kind=2, Marschner 3-lobe pdf via _hair_eval_lobes -- the same
    pdf machinery _nee_weight_hair already trusts for NEE MIS), and
    measured (mat_kind=3, tabulated-BRDF pdf via a reconstructed local
    frame). Only volume (isotropic phase, no surface normal) falls through
    to `weight=1`, today's plain unweighted behavior; dielectric/
    thin_dielectric are genuinely delta/specular and never even reach here
    at all (never stored as LVC vertices, see this file's opening VCM
    comment) -- both deliberately scoped boundaries, not silent
    omissions."""
    if cv.is_delta != Int32(0) or lv.is_delta != Int32(0):
        return (SpectralSample(Float32(0)), False)

    var d3 = lv.pos - cv.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8):
        return (SpectralSample(Float32(0)), False)
    var dist = sqrt(dist2)

    var dir = d3.to_simd() / dist
    var neg_dir = -dir

    # Both endpoints evaluate in the spectral domain and multiply there --
    # the product of two spectra, not the product of two RGB triples, which
    # is the entire point. This used to be a dual RGB/spectral path whose
    # spectral branch evaluated at the LIGHT vertex's own wavelengths and
    # discarded the camera subpath's, because LVC vertices each carried an
    # independent wavelength draw and there was no consistent basis to
    # multiply in. Every subpath in a pass now shares one wavelength set
    # (see _bdpt_pass_wavelengths), so cv and lv are guaranteed to agree and
    # the fallback is gone -- including for hair and measured, which the old
    # spectral branch had to route around.
    var wl = cv.wavelengths
    var f_cam_spec = _eval_vertex_spectral(cv, dir, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
    var f_lgt_spec: SpectralSample
    if lv.is_light == Int32(1):
        # Light emission: Le carries no cosine of its own, so the emitting
        # surface's cosine is applied HERE, explicitly -- see the geometry
        # note below for why it can no longer come from a shared G.
        var ln = lv.normal.to_simd()
        var cos_l = dot(neg_dir, ln)
        if cos_l <= Float32(0):
            return (SpectralSample(Float32(0)), False)
        f_lgt_spec = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, lv.alb.r, lv.alb.g, lv.alb.b, wl) * cos_l
    else:
        f_lgt_spec = _eval_vertex_spectral(lv, neg_dir, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
    var f_combined = f_cam_spec * f_lgt_spec

    # GEOMETRY: 1/d^2 ONLY. Each endpoint's cosine is already inside its f_cos.
    #
    # This used to be G = |cos_cv| |cos_lv| / d^2, on top of two
    # _eval_vertex_spectral values that are f*cos -- so every connection
    # applied each endpoint's surface cosine TWICE and delivered roughly half
    # its MIS share. It hid for a long time because the white-furnace cells
    # cannot see it: there the camera and light vertices lie on one plane, G
    # is ~0 and connections contribute nothing at all. The closed cavity,
    # where connections carry most of the answer, read 0.638 at the default
    # light-path count against an analytic 1.0 -- and the per-strategy split
    # showed the deficit tracking connect's own contribution almost exactly
    # (0.356 delivered / 0.35 missing at N=1024, 0.261 / 0.25 at N=4096).
    #
    # Same mistake LobeEval's docstring records merging having made (f*cos
    # where the estimator wanted bare f); this is its connection-side twin.
    # Putting the cosines in f_cos rather than dividing them back out keeps
    # the lobe's OWN cosine at each end, which is not always |cos(dir, n)| --
    # hair carries the fibre cosine and a volume none -- so this is right for
    # every lobe kind, where `f_cos / cos_used * cos*cos` would not be. It is
    # exactly the form _bdpt_connect_to_camera (t=1) already uses.
    var contrib = cv.beta * lv.beta * f_combined * (Float32(1) / dist2)

    # VCM Stage 2b/2d: real MIS weight for diffuse/conductor/light-source
    # connections (see this function's docstring + _bdpt_vertex_pdfs'/
    # project_vcm_stage2_mis_derivation memory for the full derivation and
    # its "not independently verified" caveats).
    #
    # GEOMETRIC normals here, unlike the shading normal the two f_cos above
    # were evaluated against: these are the solid-angle -> area density
    # conversion (pbrt's Vertex::ConvertDensity, which reads ng()), not a
    # BSDF cosine. A perturbed normal in a density is a bias.
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
    elif cv.is_surface == Int32(0) and lv.is_light == Int32(1) and lv.pdf_fwd > Float32(0):
        # Volume vertex -> light source: MIS against the phase-hit strategy
        # (the camera path continuing by uniform-sphere sampling and landing
        # on this emitter), whose pdf is the isotropic phase pdf 1/(4pi).
        # lv.pdf_fwd is the light point's area pdf; convert to solid angle at
        # cv. The hit side computes the same pdf in the same measure.
        var cos_lv_vol = abs(dot(neg_dir, lv.normal.to_simd()))
        if cos_lv_vol > Float32(1e-8):
            var pdf_light_w_vol = lv.pdf_fwd * dist2 / cos_lv_vol
            contrib *= power_heuristic(pdf_light_w_vol, INV_FOUR_PI)

    return (contrib, True)

# ── Main BDPT render ──────────────────────────────────────────────────────────

def _bdpt_render_core(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
    n_spp:    Int,
    n_photons_req: Int,
    verbose:  Bool,
) -> Tuple[Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]]:
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
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)
    var max_comp  = psc[unsafe_offset=0].film_max_comp
    # VCM Stage 2b: world-space size of one pixel at unit distance along the
    # camera forward axis -- same quantity the plain path tracer's mip LOD
    # uses (pipeline.mojo), reused here for the camera-origin cameraPdfW
    # derivation (see project_vcm_stage2_mis_derivation memory).
    var px_scale = Float32(2.0) * tan(psc[unsafe_offset=0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(fh)

    var n_light_paths_merge = max(n_photons_req, n_pix)
    print("VCM: " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
          + String(n_light_paths_merge) + " light paths/pass")

    var has_med = Int(sd.mediumCount) > 0

    # Determine starting medium for light subpaths (same logic as SPPM)
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[unsafe_offset=mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # Output buffer: one RGB per pixel, plus a parallel first-hit-albedo AOV
    # accumulator for the post-render denoiser (see write_image call below).
    var buf = unsafe_alloc[RGB](n_pix)
    var albedo_buf = unsafe_alloc[RGB](n_pix)
    for i in range(n_pix):
        buf[unsafe_offset=i] = RGB(Float32(0))
        albedo_buf[unsafe_offset=i] = RGB(Float32(0))

    var r2c = psc[unsafe_offset=0].raster_to_camera
    var c2w = psc[unsafe_offset=0].camera_to_world
    var base_seed = psc[unsafe_offset=0].rng_seed

    # ── t=1 light tracing: camera-projection matrices ────────────────────
    # w2c = inverse(cameraToWorld). c2r inverts the 3x3 that
    # gen_primary_ray_state uses to turn (filmX, filmY, 1) into a
    # camera-space direction -- rasterToCamera's columns 0, 1 and 3.
    var w2c = unsafe_alloc[Float32](16)
    _ = matrix_invert(c2w, w2c)
    var a0 = r2c[unsafe_offset=0]; var a1 = r2c[unsafe_offset=4]; var a2 = r2c[unsafe_offset=12]
    var b0 = r2c[unsafe_offset=1]; var b1 = r2c[unsafe_offset=5]; var b2 = r2c[unsafe_offset=13]
    var g0 = r2c[unsafe_offset=2]; var g1 = r2c[unsafe_offset=6]; var g2 = r2c[unsafe_offset=14]
    var d0 = b1*g2 - b2*g1
    var d1 = b0*g2 - b2*g0
    var d2 = b0*g1 - b1*g0
    var det = a0*d0 - a1*d1 + a2*d2
    var idet = Float32(1) / det if abs(det) > Float32(1e-20) else Float32(0)
    var c2r = unsafe_alloc[Float32](9)
    c2r[unsafe_offset=0] =  d0*idet;                 c2r[unsafe_offset=1] = -(a1*g2 - a2*g1)*idet; c2r[unsafe_offset=2] =  (a1*b2 - a2*b1)*idet
    c2r[unsafe_offset=3] = -d1*idet;                 c2r[unsafe_offset=4] =  (a0*g2 - a2*g0)*idet; c2r[unsafe_offset=5] = -(a0*b2 - a2*b0)*idet
    c2r[unsafe_offset=6] =  d2*idet;                 c2r[unsafe_offset=7] = -(a0*g1 - a1*g0)*idet; c2r[unsafe_offset=8] =  (a0*b1 - a1*b0)*idet


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
    var lvc = unsafe_alloc[BDPTVertex](max(lvc_cap, 1))
    var lvc_path_len = unsafe_alloc[Int32](max(n_light_paths_merge, 1))
    # One scratch Intersection_C per concurrent worker (light path / pixel)
    # instead of one shared slot — CPU threads now race on this exactly like
    # GPU threads already do (see _bdpt_emit_light_paths_gpu/
    # _bdpt_camera_connect_gpu's own per-thread inter_light_ptr+k/
    # inter_cam_ptr+pix), so it can no longer be a single reused buffer.
    var scratch_light = unsafe_alloc[Intersection_C](max(n_light_paths_merge, 1))
    var scratch_cam = unsafe_alloc[Intersection_C](max(n_pix, 1))

    # VCM vertex merging: grid buffers allocated once, rebuilt fresh every
    # spp sample (mirrors the LVC itself). Stage 2c: the radius itself is
    # now progressive (Hachisuka & Jensen 2008's global, per-iteration
    # scheme -- see this file's opening VCM comment for why that's the
    # right choice here, not SPPM's per-pixel adaptive radius), recomputed
    # each `si` below from `merge_radius_1`, the same initial 3%-of-scene-
    # diameter value Stage 1 used as its (then-fixed) radius.
    var (_scene_center, scene_radius) = _scene_bounding_sphere(sd)
    var merge_heads = unsafe_alloc[Int32](_HSIZE)
    var merge_next = unsafe_alloc[Int32](max(lvc_cap, 1))
    # t=1 splat records: one slot per potential light vertex.
    # Continuous raster position per splat record (x < 0 marks an empty slot)
    # -- a position, not a pixel, because the splat is spread over the filter
    # footprint at accumulation time (_bdpt_splat_filtered).
    var splat_fx = unsafe_alloc[Float32](max(n_light_paths_merge * _BDPT_MAX_VERTS, 1))
    var splat_fy = unsafe_alloc[Float32](max(n_light_paths_merge * _BDPT_MAX_VERTS, 1))
    var film_filter_cpu = film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                         psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y)
    var splat_val = unsafe_alloc[SpectralSample](max(n_light_paths_merge * _BDPT_MAX_VERTS, 1))
    var cam_pos = Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])

    for si in range(n_spp):
        # Stage 2c progressive radius: r_i = r_1 / (i+1)^(0.5*(1-alpha))
        # (Hachisuka & Jensen 2008 via Georgiev et al. 2012 Eq. 11), a
        # single GLOBAL radius shared by every pixel this sample, shrinking
        # monotonically across samples -- distinct from sppm.mojo's
        # per-pixel Knaus-Zwicker scheme (see this file's opening comment).
        var radius_i = vcm_merge_radius(scene_radius, si)
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
        # This pass's shared hero wavelengths -- see _bdpt_pass_wavelengths.
        var pass_wl = pass_wavelengths(si)

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
                                         scratch_light.unsafe_offset(lp_idx), lvc, lp_idx, lvc_path_len,
                                         mis_vc_weight_factor, mis_vm_weight_factor, pass_wl,
                                         cam_pos, px_scale)

        parallelize[emit_light_path](n_light_paths_merge)

        _bdpt_build_merge_grid(lvc, lvc_path_len, n_light_paths_merge, merge_next, merge_heads, merge_inv_cell)

        # ── Phase 1.5: t=1 light tracing (splat) ─────────────────────────
        # A light vertex lands on an ARBITRARY pixel, not the one being
        # shaded, so this cannot be folded into Phase 2's per-pixel
        # accumulation the way every other strategy is. Rather than add
        # float atomics on the film, the connections (which do the
        # expensive part -- a visibility ray each) run in parallel into a
        # per-slot record array, and the cheap accumulation is then a
        # serial pass. That also keeps the result deterministic, which
        # atomics on floats would not.
        @parameter
        def splat_light_path(lp_idx: Int):
            var base = lp_idx * _BDPT_MAX_VERTS
            for local in range(Int(lvc_path_len[unsafe_offset=lp_idx])):
                var r = _bdpt_connect_to_camera(
                    lvc[unsafe_offset=base + local], sd, scratch_light.unsafe_offset(lp_idx), cam_pos,
                    w2c, c2r, Int32(fw), Int32(fh), px_scale,
                    Float32(n_light_paths_merge), mis_vm_weight_factor)
                splat_fx[unsafe_offset=base + local] = r[1] if r[0] else Float32(-1)
                splat_fy[unsafe_offset=base + local] = r[2]
                splat_val[unsafe_offset=base + local] = r[3]
            for local in range(Int(lvc_path_len[unsafe_offset=lp_idx]), _BDPT_MAX_VERTS):
                splat_fx[unsafe_offset=base + local] = Float32(-1)

        parallelize[splat_light_path](n_light_paths_merge)

        # ── Output boundary: spectral splat -> RGB film ──────────────────
        # buf is RGB[n_pix]; the splatter takes 3 packed floats per pixel.
        comptime assert size_of[RGB]() == 3 * size_of[Float32](), "RGB must be 3 packed Float32"
        var buf_f = buf.unsafe_bitcast[Float32]()
        for k in range(n_light_paths_merge * _BDPT_MAX_VERTS):
            var sfx = splat_fx[unsafe_offset=k]
            if sfx >= Float32(0):
                var (sr, sg, sb) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, splat_val[unsafe_offset=k], lvc[unsafe_offset=k].wavelengths)
                _bdpt_splat_filtered[False](buf_f, sfx, splat_fy[unsafe_offset=k], sr, sg, sb,
                                            fw, fh, film_filter_cpu)

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
                r2c, c2w, px, py, sd, cpcg, has_med, scratch_cam.unsafe_offset(pix), lvc, pix, Int(lvc_path_len[unsafe_offset=pix]),
                merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
                px_scale, mis_vc_weight_factor, mis_vm_weight_factor, Float32(n_light_paths_merge), pass_wl,
                film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                               psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y))
            # ── Output boundary: spectral transport -> RGB film ──────────
            var (cr, cg, cb) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, contrib, pass_wl)
            buf[unsafe_offset=pix] += RGB(cr, cg, cb)
            albedo_buf[unsafe_offset=pix] += alb

        parallelize[camera_connect](n_pix)

        if verbose:
            print("VCM: sample " + String(si + 1) + "/" + String(n_spp))

    scratch_light.unsafe_free(); scratch_cam.unsafe_free(); lvc.unsafe_free(); lvc_path_len.unsafe_free()
    merge_heads.unsafe_free(); merge_next.unsafe_free()
    # Were never freed (the old splat_pix leaked the same way), once per render.
    splat_fx.unsafe_free(); splat_fy.unsafe_free(); splat_val.unsafe_free()

    # Clamp into caller-owned output buffers (no denoise/write here -- see
    # vcm_render/vcm_render_gpu, this function's two callers, for the tail).
    var inv_spp = iso_scale / Float32(n_spp)
    var pixels = unsafe_alloc[Float32](n_pix * 3)
    for i in range(n_pix):
        var c = buf[unsafe_offset=i] * inv_spp
        if max_comp > Float32(0):
            c.r = c.r if c.r < max_comp else max_comp
            c.g = c.g if c.g < max_comp else max_comp
            c.b = c.b if c.b < max_comp else max_comp
        pixels[unsafe_offset=i*3]   = c.r
        pixels[unsafe_offset=i*3+1] = c.g
        pixels[unsafe_offset=i*3+2] = c.b
    buf.unsafe_free()

    var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
    var inv_spp_alb = Float32(1) / Float32(n_spp)
    for i in range(n_pix):
        var a = albedo_buf[unsafe_offset=i] * inv_spp_alb
        albedo_pixels[unsafe_offset=i*3]   = a.r
        albedo_pixels[unsafe_offset=i*3+1] = a.g
        albedo_pixels[unsafe_offset=i*3+2] = a.b
    albedo_buf.unsafe_free()

    return Tuple[Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](pixels, albedo_pixels)

def vcm_render(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
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
    var n_pix = Int(psc[unsafe_offset=0].film_w) * Int(psc[unsafe_offset=0].film_h)
    var (pixels, albedo_pixels) = _bdpt_render_core(psc, sd, n_spp, n_photons, verbose)

    var normals = unsafe_alloc[Float32](n_pix * 3)
    var depth = unsafe_alloc[Float32](n_pix)
    var sd_local = sd
    render_aux_buffers(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world, Int32(0), Int32(0),
                        psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h, Pointer(to=sd_local), normals, depth)

    var denoised = unsafe_alloc[Float32](n_pix * 3)
    if no_denoise:
        for i in range(n_pix * 3): denoised[unsafe_offset=i] = pixels[unsafe_offset=i]
    else:
        denoise(pixels, albedo_pixels, normals, depth, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
                denoised, Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

    _ = write_image_cropwindow(denoised, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
        psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
        psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))
    pixels.unsafe_free(); albedo_pixels.unsafe_free(); normals.unsafe_free(); depth.unsafe_free(); denoised.unsafe_free()
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
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    inter_scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    n_light_paths_dp: Int64,
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx_dp: Int64,
    # camera_to_world + pixel angular size: the light subpath needs a
    # bump/normal-map footprint and has no differentials of its own, so it
    # uses the camera-distance approximation (see _bdpt_light_path_bounce).
    # The matrix, not a precomputed Vec3f, because the device already holds
    # this exact buffer for the camera kernels and only its translation
    # column is read here.
    c2w: Pointer[Float32, MutUntrackedOrigin],
    px_scale: Float32,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
    # Device-resident density fields, for the free-flight sampler (see the
    # matching comment on the bounce kernels).
    grids: Pointer[Grid_C, MutUntrackedOrigin] = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_grids: Int64 = Int64(0),
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin] = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_nvdb_grids: Int64 = Int64(0),
):
    """One thread per light path, each writing only its own dedicated
    per-path slice of `lvc` (VCM Stage 2b, see _bdpt_store_lvc_vertex's
    docstring) -- no atomics/contention. Thin wrapper: build sd, seed this
    thread's own PCG32, call the SAME _bdpt_trace_light_path vcm_render's
    CPU driver calls (with [False] on CPU, [True] here). `has_med` isn't a
    kernel parameter (`Bool` isn't a `DevicePassable` type `enqueue_function`
    accepts) -- derived here from `mediumCount`, which already is."""
    var spectral_res = Int(spectral_res_dp)
    var n_light_paths = Int(n_light_paths_dp)
    var pass_idx = Int(pass_idx_dp)
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
        grids=grids, gridCount=n_grids, nvdbGrids=nvdb_grids, nvdbGridCount=n_nvdb_grids,
    )
    var has_med = mediumCount > Int64(0)
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    var scratch = inter_scratch.unsafe_offset(k)
    var pass_wl = pass_wavelengths(pass_idx)
    _bdpt_trace_light_path[True](sd, pcg, has_med, default_emit_med, scratch, lvc, k, lvc_path_len,
                                 mis_vc_weight_factor, mis_vm_weight_factor, pass_wl,
                                 Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14]),
                                 px_scale)

def _bdpt_splat_light_paths_gpu(
    accum: Pointer[Float32, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    n_light_paths_dp: Int64,
    inter_scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    w2c: Pointer[Float32, MutUntrackedOrigin],
    c2r: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    px_scale: Float32,
    mis_vm_weight_factor: Float32,
    # The scene's PixelFilter: each splat is spread over its footprint
    # (_bdpt_splat_filtered) so the light-traced half of the image is
    # reconstructed the same way as the camera half.
    film_filter: FilmFilter,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
):
    """GPU t=1 light tracing: one thread per light path, splatting each of
    its vertices onto the film through the same `_bdpt_connect_to_camera`
    the CPU path uses, so the two backends share the estimator exactly.

    One deliberate difference from vcm_render's CPU Phase 1.5: the film is
    accumulated with float atomics here. The CPU version avoids them --
    connections are recorded into a per-slot array in parallel and summed
    in a serial pass, which keeps its film bitwise deterministic. That does
    not carry over: a light vertex lands on an ARBITRARY pixel, so unlike
    every other kernel here (each of which owns its output slot) collisions
    between threads are the normal case, and resolving them without atomics
    would need either a host readback of tens of MB per pass or a second
    scatter kernel that needs atomics anyway.

    The cost is that --gpu --vcm's film is not bitwise reproducible across
    runs, since float addition is not associative and the atomic order is
    arbitrary. Use CPU --vcm when bitwise determinism matters, e.g. when
    A/B-testing an estimator change.

    Launched on the same stream AFTER _bdpt_camera_connect_gpu so that
    kernel's non-atomic per-pixel `accum[pix*3] += ...` (race-free only
    because each of its threads owns one pixel) can never overlap these
    atomic adds."""
    var n_light_paths = Int(n_light_paths_dp)
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
        spectral_coeffs, Int(spectral_res_dp), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        measuredBrdfs, measuredBrdfCount,
        gpuTextures, gpuTextureCount,
    )
    var cam_pos = Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])
    var scratch = inter_scratch.unsafe_offset(k)
    var base = k * _BDPT_MAX_VERTS
    var n_verts = Int(lvc_path_len[unsafe_offset=k])
    for local in range(min(n_verts, _BDPT_MAX_VERTS)):
        var r = _bdpt_connect_to_camera(
            lvc[unsafe_offset=base + local], sd, scratch, cam_pos,
            w2c, c2r, Int32(Int(fw_dp)), Int32(Int(fh_dp)), px_scale,
            Float32(n_light_paths), mis_vm_weight_factor)
        if r[0]:
            var (cr, cg, cb) = spectral_sample_to_rgb(
                spectral_coeffs, Int(spectral_res_dp), spectral_cie_x, spectral_cie_y,
                spectral_cie_z, spectral_d65, r[3], lvc[unsafe_offset=base + local].wavelengths)
            _bdpt_splat_filtered[True](accum, r[1], r[2], cr, cg, cb,
                                       Int(fw_dp), Int(fh_dp), film_filter)

def _bdpt_camera_connect_gpu(
    accum: Pointer[Float32, MutUntrackedOrigin],
    albedo_accum: Pointer[Float32, MutUntrackedOrigin],
    n_pix_dp: Int64,
    fw_dp: Int64,
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    inter_scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    px_scale: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_f: Float32,
    film_filter: FilmFilter,
    seed: UInt64,
    pass_idx_dp: Int64,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
    # Device-resident density fields, for the free-flight sampler (see the
    # matching comment on the bounce kernels).
    grids: Pointer[Grid_C, MutUntrackedOrigin] = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_grids: Int64 = Int64(0),
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin] = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_nvdb_grids: Int64 = Int64(0),
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
    var spectral_res = Int(spectral_res_dp)
    var n_pix = Int(n_pix_dp)
    var fw = Int(fw_dp)
    var pass_idx = Int(pass_idx_dp)
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
        grids=grids, gridCount=n_grids, nvdbGrids=nvdb_grids, nvdbGridCount=n_nvdb_grids,
    )
    var has_med = mediumCount > Int64(0)
    var px = pix % fw
    var py = pix // fw
    var pcg = PCG32(seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                     UInt64(pass_idx * 2654435761 + 1))
    var scratch = inter_scratch.unsafe_offset(pix)
    var pass_wl = pass_wavelengths(pass_idx)
    var (contrib, alb) = _bdpt_trace_camera_and_connect[True](
        r2c, c2w, px, py, sd, pcg, has_med, scratch, lvc, pix, Int(lvc_path_len[unsafe_offset=pix]),
        merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
        px_scale, mis_vc_weight_factor, mis_vm_weight_factor, n_light_paths_f, pass_wl, film_filter)
    # ── Output boundary: spectral transport -> RGB film ──────────────────
    var (cr, cg, cb) = spectral_sample_to_rgb(
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, contrib, pass_wl)
    accum[unsafe_offset=pix*3]   += cr
    accum[unsafe_offset=pix*3+1] += cg
    accum[unsafe_offset=pix*3+2] += cb
    albedo_accum[unsafe_offset=pix*3]   += alb.r
    albedo_accum[unsafe_offset=pix*3+1] += alb.g
    albedo_accum[unsafe_offset=pix*3+2] += alb.b


# ── Task #163 stage 4, part 3: wavefront-staged GPU kernels ─────────────────
# Thin per-bounce wrappers around _bdpt_light_path_init/_bounce and
# _bdpt_camera_path_init/_bounce (this file, above), the counterpart to
# _bdpt_emit_light_paths_gpu/_bdpt_camera_connect_gpu's single-mega-kernel
# design. Each subpath type gets 3 kernels (init once, then intersect+bounce
# once per depth level, host-loop driven -- see vcm_render_gpu_wavefront's
# docstring below) plus a final accumulate kernel for the camera side (whose
# `total`/`first_alb` only get written into accum/albedo_accum once, after
# the whole subpath is done, unlike the light side which has no equivalent
# accumulator). `results`/`inter_*_ptr` intentionally double as BOTH this
# bounce's primary-ray intersection storage AND the scratch buffer
# `_bdpt_camera_path_bounce`'s internal shadow-ray probes reuse -- same
# single-scratch-slot-per-thread convention `_bdpt_camera_connect_gpu`
# already used. The intersect kernels use plain `traverse_bvh2_core`/
# `test_spheres` (NOT gpu.mojo's curve-deferred `traverse_bvh2_core_defer_
# curves`/`traverse_paths_gpu` machinery) -- matching exactly what
# `_bdpt_light_path_bounce`/`_bdpt_camera_path_bounce` themselves expect
# (a single resolved Intersection_C, no deferred-curve candidate list) and
# what the CPU-side split functions' own equivalence tests already verified
# against. This is a deliberate, first-pass scope match to
# `_bdpt_emit_light_paths_gpu`'s existing intersect behavior -- swapping
# this specific step for Vulkan RT (stages 1-3's interop mechanism) is the
# next piece of work once this staging is itself verified correct.

def _bdpt_light_path_init_gpu(
    states: Pointer[VCMLightPathState_C, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    n_light_paths_dp: Int64,
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx_dp: Int64,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
):
    """One thread per light path: seed this thread's own PCG32 (same seed
    formula _bdpt_emit_light_paths_gpu uses), call _bdpt_light_path_init,
    store the resulting VCMLightPathState_C. Mirrors
    _bdpt_emit_light_paths_gpu's docstring for why `has_med` isn't a kernel
    parameter -- not needed here since init doesn't touch media."""
    var spectral_res = Int(spectral_res_dp)
    var n_light_paths = Int(n_light_paths_dp)
    var pass_idx = Int(pass_idx_dp)
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
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    var pass_wl = pass_wavelengths(pass_idx)
    states[unsafe_offset=k] = _bdpt_light_path_init[True](sd, pcg, default_emit_med, k, lvc, lvc_path_len, mis_vc_weight_factor, pass_wl)

def _bdpt_light_path_intersect_gpu(
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    states: Pointer[VCMLightPathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Batched-per-thread primary/bounce-ray intersect for one light-path
    depth level -- separated from the material dispatch in
    _bdpt_light_path_bounce_gpu so this specific step (and only this step)
    is the eventual Vulkan RT swap point, matching the plain wavefront path
    tracer's traverse_paths_gpu/shade_*_gpu split."""
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if states[unsafe_offset=tid].active == Int8(0):
        return
    var ray = Ray_C(states[unsafe_offset=tid].ro, states[unsafe_offset=tid].rd)
    results[unsafe_offset=tid].hit = Int8(0)
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, Float32(1e38), results.unsafe_offset(tid),
                        blasNodesArr, blasPrimIdsArr, instances)
    test_spheres(spheres, n_spheres, ray, results.unsafe_offset(tid))

def _bdpt_light_path_bounce_gpu(
    states: Pointer[VCMLightPathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_dp: Int64,
    # Bump/normal-map footprint reference -- see _bdpt_emit_light_paths_gpu's
    # matching params (this is the wavefront-staged path to the same walk).
    c2w: Pointer[Float32, MutUntrackedOrigin],
    px_scale: Float32,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
    # Device-resident density fields, for the free-flight sampler. Without
    # these the descriptor built below reports no density fields and every
    # heterogeneous medium samples as uniform density-1 fog.
    grids: Pointer[Grid_C, MutUntrackedOrigin] = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_grids: Int64 = Int64(0),
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin] = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_nvdb_grids: Int64 = Int64(0),
):
    """One bounce's material dispatch for one light path, reading the
    Intersection_C _bdpt_light_path_intersect_gpu already computed this
    depth level instead of tracing it inline -- see this section's opening
    comment."""
    var spectral_res = Int(spectral_res_dp)
    var n_light_paths = Int(n_light_paths_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_light_paths:
        return
    if states[unsafe_offset=k].active == Int8(0):
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
        grids=grids, gridCount=n_grids, nvdbGrids=nvdb_grids, nvdbGridCount=n_nvdb_grids,
    )
    var has_med = mediumCount > Int64(0)
    var pcg = PCG32(UInt64(0), UInt64(0))
    pcg.state = states[unsafe_offset=k].pcg_state
    pcg.inc = states[unsafe_offset=k].pcg_inc
    var ro = states[unsafe_offset=k].ro
    var rd = states[unsafe_offset=k].rd
    var flux = states[unsafe_offset=k].flux
    var n_verts = Int(states[unsafe_offset=k].n_verts)
    var dvcm_carry = states[unsafe_offset=k].dvcm
    var dvc_carry = states[unsafe_offset=k].dvc
    var dvm_carry = states[unsafe_offset=k].dvm
    var is_finite_origin = states[unsafe_offset=k].is_finite_origin == Int8(1)
    var cur_med_idx = states[unsafe_offset=k].cur_med_idx
    var n_lbounces = Int(states[unsafe_offset=k].n_lbounces)
    var current_dielectric_ior = states[unsafe_offset=k].current_dielectric_ior
    var previous_dielectric_ior = states[unsafe_offset=k].previous_dielectric_ior
    var wavelengths = SampledWavelengths(states[unsafe_offset=k].wl0, states[unsafe_offset=k].wl1, states[unsafe_offset=k].wl2, states[unsafe_offset=k].wl3, states[unsafe_offset=k].wl_pdf)

    var cont = _bdpt_light_path_bounce[True](
        sd, pcg, has_med, results[unsafe_offset=k], lvc, k, mis_vc_weight_factor, mis_vm_weight_factor,
        ro, rd, flux, n_verts, dvcm_carry, dvc_carry, dvm_carry,
        is_finite_origin, cur_med_idx, n_lbounces,
        current_dielectric_ior, previous_dielectric_ior, wavelengths,
        Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14]), px_scale,
    )
    lvc_path_len[unsafe_offset=k] = Int32(n_verts)
    states[unsafe_offset=k].active = Int8(1) if cont else Int8(0)
    states[unsafe_offset=k].ro = ro
    states[unsafe_offset=k].rd = rd
    states[unsafe_offset=k].flux = flux
    states[unsafe_offset=k].n_verts = Int32(n_verts)
    states[unsafe_offset=k].dvcm = dvcm_carry
    states[unsafe_offset=k].dvc = dvc_carry
    states[unsafe_offset=k].dvm = dvm_carry
    states[unsafe_offset=k].cur_med_idx = cur_med_idx
    states[unsafe_offset=k].n_lbounces = Int32(n_lbounces)
    states[unsafe_offset=k].current_dielectric_ior = current_dielectric_ior
    states[unsafe_offset=k].previous_dielectric_ior = previous_dielectric_ior
    states[unsafe_offset=k].pcg_state = pcg.state
    states[unsafe_offset=k].pcg_inc = pcg.inc

def _bdpt_camera_path_init_gpu(
    states: Pointer[VCMCameraPathState_C, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    n_pix_dp: Int64,
    fw_dp: Int64,
    px_scale: Float32,
    n_light_paths_f: Float32,
    film_filter: FilmFilter,
    seed: UInt64,
    pass_idx_dp: Int64,
):
    """One thread per pixel: seed this thread's own PCG32 (same seed formula
    _bdpt_camera_connect_gpu uses), call _bdpt_camera_path_init, store the
    resulting VCMCameraPathState_C. No scene params needed -- camera-ray
    generation doesn't touch the scene."""
    var n_pix = Int(n_pix_dp)
    var fw = Int(fw_dp)
    var pass_idx = Int(pass_idx_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var px = pix % fw
    var py = pix // fw
    var pcg = PCG32(seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                     UInt64(pass_idx * 2654435761 + 1))
    var pass_wl = pass_wavelengths(pass_idx)
    states[unsafe_offset=pix] = _bdpt_camera_path_init[True](r2c, c2w, px, py, pcg, px_scale, n_light_paths_f, pass_wl, film_filter)

def _bdpt_camera_path_intersect_gpu(
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    states: Pointer[VCMCameraPathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Camera-path counterpart to _bdpt_light_path_intersect_gpu -- see its
    docstring."""
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if states[unsafe_offset=tid].active == Int8(0):
        return
    var ray = Ray_C(states[unsafe_offset=tid].ro, states[unsafe_offset=tid].rd)
    results[unsafe_offset=tid].hit = Int8(0)
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, Float32(1e38), results.unsafe_offset(tid),
                        blasNodesArr, blasPrimIdsArr, instances)
    test_spheres(spheres, n_spheres, ray, results.unsafe_offset(tid))

def _bdpt_camera_path_bounce_gpu(
    states: Pointer[VCMCameraPathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_pix_dp: Int64,
    # Bump/normal-map footprint reference -- see _bdpt_camera_path_bounce's
    # matching params.
    c2w: Pointer[Float32, MutUntrackedOrigin],
    px_scale: Float32,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
    # Task #163 stage 5: see _bdpt_camera_path_bounce's own matching
    # params -- forwarded through unchanged, except Bool -> Int8 (raw kernel
    # launch args must be DevicePassable; Bool doesn't conform, unlike a
    # Bool that's merely a regular-function parameter one level down).
    defer_shadow_rays: Int8 = Int8(0),
    shadow_rays: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin] = Pointer[SpectralSample, MutUntrackedOrigin].unsafe_dangling(),
    shadow_valid: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    # Device-resident density fields, for the free-flight sampler. Without
    # these the descriptor built below reports no density fields and every
    # heterogeneous medium samples as uniform density-1 fog.
    grids: Pointer[Grid_C, MutUntrackedOrigin] = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_grids: Int64 = Int64(0),
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin] = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_nvdb_grids: Int64 = Int64(0),
):
    """One bounce's material dispatch (incl. NEE/connect/merge/MNEE, all
    still on the existing software-BVH `results + pix` scratch slot -- see
    this section's opening comment) for one camera path, reading the
    Intersection_C _bdpt_camera_path_intersect_gpu already computed this
    depth level."""
    var spectral_res = Int(spectral_res_dp)
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    if states[unsafe_offset=pix].active == Int8(0):
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
        grids=grids, gridCount=n_grids, nvdbGrids=nvdb_grids, nvdbGridCount=n_nvdb_grids,
    )
    var has_med = mediumCount > Int64(0)
    var pcg = PCG32(UInt64(0), UInt64(0))
    pcg.state = states[unsafe_offset=pix].pcg_state
    pcg.inc = states[unsafe_offset=pix].pcg_inc
    var ro = states[unsafe_offset=pix].ro
    var rd = states[unsafe_offset=pix].rd
    var beta = states[unsafe_offset=pix].beta
    var total = states[unsafe_offset=pix].total
    var first_alb = states[unsafe_offset=pix].first_alb
    var n_verts = Int(states[unsafe_offset=pix].n_verts)
    var n_bounces = Int(states[unsafe_offset=pix].n_bounces)
    var cur_med_idx = states[unsafe_offset=pix].cur_med_idx
    var dvcm_carry = states[unsafe_offset=pix].dvcm
    var dvc_carry = states[unsafe_offset=pix].dvc
    var dvm_carry = states[unsafe_offset=pix].dvm
    var last_bsdf_pdf = states[unsafe_offset=pix].last_bsdf_pdf
    var mis_null_dist = states[unsafe_offset=pix].mis_null_dist
    var current_dielectric_ior = states[unsafe_offset=pix].current_dielectric_ior
    var previous_dielectric_ior = states[unsafe_offset=pix].previous_dielectric_ior
    var wavelengths = SampledWavelengths(states[unsafe_offset=pix].wl0, states[unsafe_offset=pix].wl1, states[unsafe_offset=pix].wl2, states[unsafe_offset=pix].wl3, states[unsafe_offset=pix].wl_pdf)

    var cont = _bdpt_camera_path_bounce[True](
        sd, pcg, has_med, results[unsafe_offset=pix], results.unsafe_offset(pix), lvc, pix, Int(lvc_path_len[unsafe_offset=pix]),
        merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
        mis_vc_weight_factor, mis_vm_weight_factor,
        ro, rd, beta, total, first_alb, n_verts, n_bounces, cur_med_idx,
        dvcm_carry, dvc_carry, dvm_carry, last_bsdf_pdf, mis_null_dist,
        current_dielectric_ior, previous_dielectric_ior, wavelengths,
        Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14]), px_scale,
        defer_shadow_rays != Int8(0), shadow_rays, shadow_pending, shadow_valid, shadow_seg_med,
    )
    states[unsafe_offset=pix].active = Int8(1) if cont else Int8(0)
    states[unsafe_offset=pix].ro = ro
    states[unsafe_offset=pix].rd = rd
    states[unsafe_offset=pix].beta = beta
    states[unsafe_offset=pix].total = total
    states[unsafe_offset=pix].first_alb = first_alb
    states[unsafe_offset=pix].n_verts = Int32(n_verts)
    states[unsafe_offset=pix].n_bounces = Int32(n_bounces)
    states[unsafe_offset=pix].cur_med_idx = cur_med_idx
    states[unsafe_offset=pix].dvcm = dvcm_carry
    states[unsafe_offset=pix].dvc = dvc_carry
    states[unsafe_offset=pix].dvm = dvm_carry
    states[unsafe_offset=pix].last_bsdf_pdf = last_bsdf_pdf
    states[unsafe_offset=pix].mis_null_dist = mis_null_dist
    states[unsafe_offset=pix].current_dielectric_ior = current_dielectric_ior
    states[unsafe_offset=pix].previous_dielectric_ior = previous_dielectric_ior
    states[unsafe_offset=pix].pcg_state = pcg.state
    states[unsafe_offset=pix].pcg_inc = pcg.inc

def _bdpt_camera_path_accumulate_gpu(
    states: Pointer[VCMCameraPathState_C, MutUntrackedOrigin],
    accum: Pointer[Float32, MutUntrackedOrigin],
    albedo_accum: Pointer[Float32, MutUntrackedOrigin],
    n_pix_dp: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin],
    spectral_res_dp: Int64,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
):
    """Runs once per `si` sample, after the camera-path bounce loop has
    fully terminated for every lane -- writes each pixel's now-complete
    `total`/`first_alb` (accumulated across every bounce inside the state
    struct) into the persistent per-pixel accum buffers, exactly once,
    matching what _bdpt_camera_connect_gpu's own single `accum[...] +=
    contrib...` did at the end of its one-shot whole-subpath trace."""
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    # ── Output boundary: spectral transport -> RGB film ──────────────────
    var wl_acc = SampledWavelengths(states[unsafe_offset=pix].wl0, states[unsafe_offset=pix].wl1,
                                    states[unsafe_offset=pix].wl2, states[unsafe_offset=pix].wl3, states[unsafe_offset=pix].wl_pdf)
    var (tr, tg, tb) = spectral_sample_to_rgb(
        spectral_coeffs, Int(spectral_res_dp), spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, states[unsafe_offset=pix].total, wl_acc)
    accum[unsafe_offset=pix*3]   += tr
    accum[unsafe_offset=pix*3+1] += tg
    accum[unsafe_offset=pix*3+2] += tb
    albedo_accum[unsafe_offset=pix*3]   += states[unsafe_offset=pix].first_alb.r
    albedo_accum[unsafe_offset=pix*3+1] += states[unsafe_offset=pix].first_alb.g
    albedo_accum[unsafe_offset=pix*3+2] += states[unsafe_offset=pix].first_alb.b

# ── Task #163 stage 4 part 4: Vulkan RT interop intersect for VCM ───────────
# Swaps _bdpt_light_path_intersect_gpu/_bdpt_camera_path_intersect_gpu's
# software-BVH traverse_bvh2_core/test_spheres for the stage-1/2/3 CUDA/
# Vulkan interop mechanism (see project_vulkan_rt_backend memory) --
# real GPU-side ray-query tracing through shared CUDA/Vulkan memory, no
# CPU round trip. Mirrors gpu.mojo's vulkaninterop_pack_rays_kernel/
# vulkaninterop_rt_traverse_paths_gpu exactly, just reading ro/rd from
# VCMLightPathState_C/VCMCameraPathState_C instead of PathState_C.ray --
# vulkaninterop_unpack_results_kernel itself is reused UNCHANGED from
# gpu.mojo for both (its output is always a plain Intersection_C, with no
# dependency on which subpath produced the ray). Scope: triangle geometry
# only, matching vulkaninterop_rt_create_scene -- callers must only use
# this for scenes with no curves/spheres/object instancing (same boundary
# debug_render_vulkanrt/--vulkan-rt-shade already enforce).

def vulkaninterop_pack_light_rays_kernel(
    states: Pointer[VCMLightPathState_C, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ro = states[unsafe_offset=tid].ro
    var rd = states[unsafe_offset=tid].rd
    var idx = tid * 8
    rays[unsafe_offset=idx + 0] = ro.x
    rays[unsafe_offset=idx + 1] = ro.y
    rays[unsafe_offset=idx + 2] = ro.z
    rays[unsafe_offset=idx + 3] = Float32(1e-4)
    rays[unsafe_offset=idx + 4] = rd.x
    rays[unsafe_offset=idx + 5] = rd.y
    rays[unsafe_offset=idx + 6] = rd.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8)

def vulkaninterop_pack_camera_rays_kernel(
    states: Pointer[VCMCameraPathState_C, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ro = states[unsafe_offset=tid].ro
    var rd = states[unsafe_offset=tid].rd
    var idx = tid * 8
    rays[unsafe_offset=idx + 0] = ro.x
    rays[unsafe_offset=idx + 1] = ro.y
    rays[unsafe_offset=idx + 2] = ro.z
    rays[unsafe_offset=idx + 3] = Float32(1e-4)
    rays[unsafe_offset=idx + 4] = rd.x
    rays[unsafe_offset=idx + 5] = rd.y
    rays[unsafe_offset=idx + 6] = rd.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8)

def vulkaninterop_rt_traverse_light_paths_gpu(
    ctx: DeviceContext,
    state_buf: DeviceBuffer[DType.uint8],
    inter_buf: DeviceBuffer[DType.uint8],
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    interop_results_buf: DeviceBuffer[DType.float32],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)

    ctx.enqueue_function[vulkaninterop_pack_light_rays_kernel](
        state_buf.unsafe_ptr().unsafe_bitcast[VCMLightPathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_total),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)

    # GPU kernel dispatch has no default-argument fill-in (unlike ordinary
    # Mojo calls) -- must pass instance_base_mesh explicitly. VCM's Vulkan
    # RT path doesn't support object instancing (its own pipeline.mojo call
    # site keeps the pre-existing instance_count>0 CUDA-fallback guard), so
    # this is always the inert dangling default.
    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        interop_results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_meshes),
        Int64(n_total),
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
        grid_dim=grid, block_dim=block_size,
    )

def vulkaninterop_rt_traverse_camera_paths_gpu(
    ctx: DeviceContext,
    state_buf: DeviceBuffer[DType.uint8],
    inter_buf: DeviceBuffer[DType.uint8],
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    interop_results_buf: DeviceBuffer[DType.float32],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)

    ctx.enqueue_function[vulkaninterop_pack_camera_rays_kernel](
        state_buf.unsafe_ptr().unsafe_bitcast[VCMCameraPathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_total),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)

    # GPU kernel dispatch has no default-argument fill-in (unlike ordinary
    # Mojo calls) -- must pass instance_base_mesh explicitly. VCM's Vulkan
    # RT path doesn't support object instancing (its own pipeline.mojo call
    # site keeps the pre-existing instance_count>0 CUDA-fallback guard), so
    # this is always the inert dangling default.
    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        interop_results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_meshes),
        Int64(n_total),
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
        grid_dim=grid, block_dim=block_size,
    )

def vulkaninterop_pack_all_shadow_rays_kernel(
    # Perf (2026-07-13, task #163 stage 5 follow-up): packs ALL
    # n_pix*_BDPT_MAX_VERTS shadow-ray slots in ONE dispatch instead of
    # _BDPT_MAX_VERTS(10) separate n_pix-sized ones -- the interop scene's
    # ray capacity was resized (pipeline.mojo) specifically to make this
    # possible, replacing the earlier per-local-slot loop. `shadow_rays`
    # and `out_rays` share the EXACT SAME idx=pix*_BDPT_MAX_VERTS+local
    # indexing scheme (both strided _BDPT_MAX_VERTS per pixel), so this is
    # a straight elementwise copy, not a re-layout. Invalid slots get a
    # zero-length degenerate ray (tMax=0) so the trace call never reads
    # uninitialized memory -- resolve skips them via shadow_valid regardless.
    shadow_rays: Pointer[Float32, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    out_rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx8 = tid * 8
    if shadow_valid[unsafe_offset=tid] == Int8(0):
        out_rays[unsafe_offset=idx8 + 0] = Float32(0)
        out_rays[unsafe_offset=idx8 + 1] = Float32(0)
        out_rays[unsafe_offset=idx8 + 2] = Float32(0)
        out_rays[unsafe_offset=idx8 + 3] = Float32(0)
        out_rays[unsafe_offset=idx8 + 4] = Float32(0)
        out_rays[unsafe_offset=idx8 + 5] = Float32(0)
        out_rays[unsafe_offset=idx8 + 6] = Float32(1)
        out_rays[unsafe_offset=idx8 + 7] = Float32(0)
        return
    out_rays[unsafe_offset=idx8 + 0] = shadow_rays[unsafe_offset=idx8 + 0]
    out_rays[unsafe_offset=idx8 + 1] = shadow_rays[unsafe_offset=idx8 + 1]
    out_rays[unsafe_offset=idx8 + 2] = shadow_rays[unsafe_offset=idx8 + 2]
    out_rays[unsafe_offset=idx8 + 3] = shadow_rays[unsafe_offset=idx8 + 3]
    out_rays[unsafe_offset=idx8 + 4] = shadow_rays[unsafe_offset=idx8 + 4]
    out_rays[unsafe_offset=idx8 + 5] = shadow_rays[unsafe_offset=idx8 + 5]
    out_rays[unsafe_offset=idx8 + 6] = shadow_rays[unsafe_offset=idx8 + 6]
    out_rays[unsafe_offset=idx8 + 7] = shadow_rays[unsafe_offset=idx8 + 7]

def vulkaninterop_rt_traverse_shadow_gpu(
    # Perf (2026-07-13): pack -> trace only, no unpack step -- the caller's
    # resolve_shadow_connect_gpu reads interop_results_buf's raw float
    # layout directly (it only needs hit/material-index, not a full
    # reconstructed Intersection_C). ONE dispatch over ALL
    # n_pix*_BDPT_MAX_VERTS shadow-ray slots (see
    # vulkaninterop_pack_all_shadow_rays_kernel above) -- requires the
    # interop scene's ray capacity to have been sized for that (see
    # pipeline.mojo's max_rays_vk_vcm), not just n_pix.
    ctx: DeviceContext,
    shadow_rays_buf: DeviceBuffer[DType.uint8],
    shadow_valid_buf: DeviceBuffer[DType.uint8],
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    count: Int,
) raises:
    comptime block_size = 256
    var grid = ceildiv(count, block_size)

    ctx.enqueue_function[vulkaninterop_pack_all_shadow_rays_kernel](
        shadow_rays_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        shadow_valid_buf.unsafe_ptr().unsafe_bitcast[Int8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(count),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(count), cuda_stream)

def bdpt_merge_grid_reset_gpu(heads: Pointer[Int32, MutUntrackedOrigin], hsize_dp: Int64):
    var hsize = Int(hsize_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= hsize:
        return
    _bdpt_reset_merge_cell(heads, tid)


def bdpt_merge_grid_insert_gpu(
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    lvc_cap_dp: Int64,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    var lvc_cap = Int(lvc_cap_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= lvc_cap:
        return
    _bdpt_insert_merge_vertex[True](k, lvc, lvc_path_len, merge_next, heads, inv_cell)


# ── Host driver ───────────────────────────────────────────────────────────

def vcm_render_gpu(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
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
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)
    var max_comp  = psc[unsafe_offset=0].film_max_comp
    var n_light_paths_merge = max(n_photons_req, n_pix)

    print("VCM (GPU): " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
          + String(n_light_paths_merge) + " light paths/pass")

    var has_med = Int(sd.mediumCount) > 0
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[unsafe_offset=mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    var lvc_cap = n_light_paths_merge * _BDPT_MAX_VERTS
    var base_seed = psc[unsafe_offset=0].rng_seed

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
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)
            var albedo_accum_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with albedo_accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)

            var r2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with r2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].raster_to_camera.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            var c2w_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with c2w_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].camera_to_world.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            # t=1 light tracing needs the same two camera-projection
            # matrices vcm_render builds on the CPU (see its matching
            # comment): w2c = inverse(cameraToWorld), and c2r inverting the
            # 3x3 that turns (filmX, filmY, 1) into a camera-space direction.
            var w2c_host = unsafe_alloc[Float32](16)
            _ = matrix_invert(psc[unsafe_offset=0].camera_to_world, w2c_host)
            var _r2c_h = psc[unsafe_offset=0].raster_to_camera
            var a0 = _r2c_h[unsafe_offset=0]; var a1 = _r2c_h[unsafe_offset=4]; var a2 = _r2c_h[unsafe_offset=12]
            var b0 = _r2c_h[unsafe_offset=1]; var b1 = _r2c_h[unsafe_offset=5]; var b2 = _r2c_h[unsafe_offset=13]
            var g0 = _r2c_h[unsafe_offset=2]; var g1 = _r2c_h[unsafe_offset=6]; var g2 = _r2c_h[unsafe_offset=14]
            var d0 = b1*g2 - b2*g1
            var d1 = b0*g2 - b2*g0
            var d2 = b0*g1 - b1*g0
            var det = a0*d0 - a1*d1 + a2*d2
            var idet = Float32(1) / det if abs(det) > Float32(1e-20) else Float32(0)
            var c2r_host = unsafe_alloc[Float32](9)
            c2r_host[unsafe_offset=0] =  d0*idet; c2r_host[unsafe_offset=1] = -(a1*g2 - a2*g1)*idet; c2r_host[unsafe_offset=2] =  (a1*b2 - a2*b1)*idet
            c2r_host[unsafe_offset=3] = -d1*idet; c2r_host[unsafe_offset=4] =  (a0*g2 - a2*g0)*idet; c2r_host[unsafe_offset=5] = -(a0*b2 - a2*b0)*idet
            c2r_host[unsafe_offset=6] =  d2*idet; c2r_host[unsafe_offset=7] = -(a0*g1 - a1*g0)*idet; c2r_host[unsafe_offset=8] =  (a0*b1 - a1*b0)*idet
            var w2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with w2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = w2c_host.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            var c2r_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](9 * size_of[Float32]())
            with c2r_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = c2r_host.unsafe_bitcast[UInt8]()
                for i in range(9 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            w2c_host.unsafe_free()
            c2r_host.unsafe_free()
            var w2c_ptr = w2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2r_ptr = c2r_buf.unsafe_ptr().unsafe_bitcast[Float32]()

            var lvc_ptr     = lvc_buf.unsafe_ptr().unsafe_bitcast[BDPTVertex]()
            var path_len_ptr = path_len_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_heads_ptr = merge_heads_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_next_ptr  = merge_next_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var inter_light_ptr = inter_light_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]()
            var inter_cam_ptr   = inter_cam_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]()
            var accum_ptr   = accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var albedo_accum_ptr = albedo_accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]()

            var bvh2Nodes = handle[].bvh.nodes_ptr()
            var primIds = handle[].bvh.prim_ids_ptr()
            var meshes = handle[].meshes.meshes_ptr()
            var curves = handle[].curves.curves_ptr()
            var blasNodesArr = handle[].blas.nodes_arr()
            var blasPrimIdsArr = handle[].blas.primids_arr()
            var instances = handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C]()
            var materials = handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C]()
            var mediums = handle[].mediums_buf.unsafe_ptr().unsafe_bitcast[Medium_C]()
            # Device-resident density fields for the free-flight sampler. These
            # were never handed to the VCM/SPPM kernels before, which is exactly
            # why those integrators sampled every heterogeneous medium as uniform
            # density-1 fog -- see geometry.mojo's sample_free_flight.
            var grids_dev = handle[].grids_buf.unsafe_ptr().unsafe_bitcast[Grid_C]()
            var nvdb_grids_dev = handle[].nvdb_grids_buf.unsafe_ptr().unsafe_bitcast[NvdbGrid_C]()
            var mediumInterfaces = handle[].medium_ifaces_buf.unsafe_ptr().unsafe_bitcast[MediumInterface_C]()
            var spheres = handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C]()
            var areaLights = handle[].lights.area_lights_ptr()
            var distantLights = handle[].lights.distant_lights_ptr()
            var infiniteLights = handle[].lights.infinite_lights_ptr()
            var pointLights = handle[].lights.point_lights_ptr()
            var n_mediums = Int64(handle[].n_mediums)
            var n_medium_ifaces = Int64(handle[].n_medium_ifaces)
            var n_spheres = Int64(handle[].n_spheres)
            var n_curves = Int64(handle[].curves.n_curves)
            var n_area_lights = Int64(handle[].lights.n_area_lights)
            var n_distant_lights = Int64(handle[].lights.n_distant_lights)
            var n_infinite_lights = Int64(handle[].lights.n_infinite_lights)
            var n_point_lights = Int64(handle[].lights.n_point_lights)
            var n_blas = Int64(handle[].blas.n_blas)
            var n_instances = Int64(handle[].n_instances)
            var (spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) = handle[].spectral.unsafe_ptrs()
            var measured_brdfs = handle[].measured_brdfs_buf.unsafe_ptr().unsafe_bitcast[MeasuredBRDF_C]()
            var n_measured_brdfs = Int64(handle[].n_measured_brdfs)
            var gpu_textures = handle[].textures.textures_ptr()
            var n_gpu_textures = Int64(handle[].textures.n_textures)

            var grid_light = ceildiv(max(n_light_paths_merge, 1), block_size)
            var grid_pix = ceildiv(n_pix, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            var (_scene_center, scene_radius) = _scene_bounding_sphere(sd)
            var px_scale = Float32(2.0) * tan(psc[unsafe_offset=0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(fh)
            var n_light_paths_f = Float32(n_light_paths_merge)

            var grid_merge_ins = ceildiv(max(lvc_cap, 1), block_size)

            for si in range(n_spp):
                # Stage 2c progressive radius -- see vcm_render (CPU)'s
                # matching per-sample loop for the full derivation comment.
                var radius_i = vcm_merge_radius(scene_radius, si)
                var merge_r2 = radius_i * radius_i
                var merge_inv_cell = Float32(1.0) / max(radius_i, Float32(1e-6))
                var merge_norm = Float32(1.0) / (Float32(n_light_paths_merge) * PI * max(merge_r2, Float32(1e-12)))
                var eta_vcm = PI * max(merge_r2, Float32(1e-12)) * Float32(n_light_paths_merge)
                var mis_vm_weight_factor = eta_vcm
                var mis_vc_weight_factor = Float32(1.0) / eta_vcm

                var pass_seed = base_seed ^ UInt64(si * 2654435761 + 1)
                handle[].ctx.enqueue_function[_bdpt_emit_light_paths_gpu](
                    lvc_ptr, path_len_ptr, mis_vc_weight_factor, mis_vm_weight_factor,
                    inter_light_ptr, Int64(n_light_paths_merge),
                    default_emit_med, pass_seed, Int64(si),
                    c2w_ptr, px_scale,
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grids_dev, Int64(handle[].n_grids), nvdb_grids_dev, Int64(handle[].n_nvdb_grids),
                    grid_dim=grid_light, block_dim=block_size)

                # VCM Stage 2b: light paths are deterministically paired with
                # pixels (the first n_pix of n_light_paths_merge total light
                # paths, task #152), so no host readback of a total vertex
                # count is needed anymore -- lvc_cap is already known at
                # compile/host time. Kernels stay ordered on one stream
                # without an explicit synchronize() here.
                handle[].ctx.enqueue_function[bdpt_merge_grid_reset_gpu](
                    merge_heads_ptr, Int64(_HSIZE), grid_dim=grid_hsize, block_dim=block_size)
                handle[].ctx.enqueue_function[bdpt_merge_grid_insert_gpu](
                    lvc_ptr, path_len_ptr, Int64(lvc_cap), merge_next_ptr, merge_heads_ptr, merge_inv_cell,
                    grid_dim=grid_merge_ins, block_dim=block_size)

                handle[].ctx.enqueue_function[_bdpt_camera_connect_gpu](
                    accum_ptr, albedo_accum_ptr, Int64(n_pix), Int64(Int(psc[unsafe_offset=0].film_w)), r2c_ptr, c2w_ptr, inter_cam_ptr,
                    lvc_ptr, path_len_ptr,
                    merge_next_ptr, merge_heads_ptr, merge_inv_cell, merge_r2, merge_norm,
                    px_scale, mis_vc_weight_factor, mis_vm_weight_factor, n_light_paths_f,
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    base_seed, Int64(si),
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grids_dev, Int64(handle[].n_grids), nvdb_grids_dev, Int64(handle[].n_nvdb_grids),
                    grid_dim=grid_pix, block_dim=block_size)

                # Phase 1.5: t=1 light tracing, the GPU counterpart of
                # vcm_render's splat pass. Same stream, launched AFTER the
                # camera connect -- see the kernel's docstring for why the
                # ordering matters and why this one uses atomics where the
                # CPU path deliberately does not.
                handle[].ctx.enqueue_function[_bdpt_splat_light_paths_gpu](
                    accum_ptr, lvc_ptr, path_len_ptr, Int64(n_light_paths_merge),
                    inter_light_ptr, w2c_ptr, c2r_ptr, c2w_ptr,
                    Int64(fw), Int64(fh), px_scale, mis_vm_weight_factor,
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grid_dim=grid_light, block_dim=block_size)

                if verbose:
                    print("VCM (GPU): sample " + String(si + 1) + "/" + String(n_spp))

            handle[].ctx.synchronize()

            var pixels = unsafe_alloc[Float32](n_pix * 3)
            with accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                var inv_spp = iso_scale / Float32(n_spp)
                for i in range(n_pix):
                    var r = src[unsafe_offset=i*3]   * inv_spp
                    var g = src[unsafe_offset=i*3+1] * inv_spp
                    var b = src[unsafe_offset=i*3+2] * inv_spp
                    if max_comp > Float32(0):
                        r = r if r < max_comp else max_comp
                        g = g if g < max_comp else max_comp
                        b = b if b < max_comp else max_comp
                    pixels[unsafe_offset=i*3] = r; pixels[unsafe_offset=i*3+1] = g; pixels[unsafe_offset=i*3+2] = b

            # Denoise (never wired up before -- no_denoise was a dead
            # parameter): read back the albedo AOV accumulated above, run
            # a fresh normals/depth pass via the host-side sd (same
            # render_aux_buffers the CPU path/plain tracer use -- host-only,
            # so it runs on the CPU here too, not as a GPU kernel), then the
            # same CPU denoise() the CPU BDPT path uses.
            var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
            with albedo_accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                var inv_spp_alb = Float32(1) / Float32(n_spp)
                for i in range(n_pix * 3):
                    albedo_pixels[unsafe_offset=i] = src[unsafe_offset=i] * inv_spp_alb

            var normals = unsafe_alloc[Float32](n_pix * 3)
            var depth = unsafe_alloc[Float32](n_pix)
            var sd_local = sd
            render_aux_buffers(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world, Int32(0), Int32(0),
                                psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h, Pointer(to=sd_local), normals, depth)

            var denoised = unsafe_alloc[Float32](n_pix * 3)
            if no_denoise:
                for i in range(n_pix * 3): denoised[unsafe_offset=i] = pixels[unsafe_offset=i]
            else:
                denoise(pixels, albedo_pixels, normals, depth, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
                        denoised, Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

            _ = write_image_cropwindow(denoised, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
        psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
        psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))
            pixels.unsafe_free(); albedo_pixels.unsafe_free(); normals.unsafe_free(); depth.unsafe_free(); denoised.unsafe_free()
        except e:
            print("VCM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("VCM GPU: no accelerator")
        ret = Int32(-1)
    return ret

# ── Task #163 stage 5: Vulkan-RT-batched shadow ray resolution ──────────────
# Resolves the diffuse-branch connect shadow rays queued by
# _bdpt_connect_to_cache_deferred: after a batched Vulkan RT dispatch fills
# shadow_inter (the SAME vulkaninterop_unpack_results_kernel used for
# primary/light/camera rays, reused unchanged), this kernel turns each hit/
# miss into a resolved Tr multiplied into shadow_pending. FAST PATH (no
# medium, and either a miss or a hit on an opaque material): resolved
# directly from the single closest-hit query, no further ray casts needed.
# FALLBACK PATH (a medium is active on this segment, or the hit is on a
# dielectric/thin_dielectric/interface material -- i.e. exactly the cases
# _visible_transmittance's own multi-round loop exists for): calls
# _visible_transmittance UNCHANGED, per-thread, software-BVH -- 100%
# correct for every scene, just not batched for these rarer rays. Neither
# of this task's two test scenes (cornell-box, dragon) has any dielectric
# material or medium, so the fallback path is written for correctness but
# not exercised by them -- see project_vulkan_rt_backend memory.

def reset_shadow_valid_gpu(
    # Task #163 stage 5: zeroes EVERY pixel's shadow_valid slots, every
    # bounce, unconditionally -- including inactive-path and non-diffuse-
    # branch pixels that _bdpt_connect_to_cache_deferred never touches this
    # bounce. Without this, a slot left valid=1 by an earlier diffuse-branch
    # bounce keeps getting re-resolved and re-summed into states[pix].total
    # on every subsequent bounce for the rest of the render (once a path
    # goes inactive, or takes a non-diffuse branch, nothing else would ever
    # clear it) -- a real overcounting bug caught via a 128spp cornell-box
    # A/B mean-radiance comparison against the software-BVH baseline (the
    # gap grew from +11% at 16spp to +62% at 128spp, the signature of a
    # per-bounce accumulating bug, not RNG-path noise). Gating this reset on
    # `states[pix].active` (cheaper, one fewer full-n_pix kernel launch)
    # does NOT work: a pixel that goes inactive DURING bounce i must still
    # keep bounce i's freshly-deferred contribution summed once, but an
    # active-gated reset can't tell "just went inactive this bounce" (keep)
    # apart from "went inactive last bounce" (must now read as all-invalid)
    # -- both read `active=0` by the time this would run. An unconditional
    # per-bounce reset sidesteps that distinction entirely.
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    n_pix_dp: Int64,
):
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var base = pix * _BDPT_MAX_VERTS
    for local in range(_BDPT_MAX_VERTS):
        shadow_valid[unsafe_offset=base + local] = Int8(0)

def resolve_shadow_connect_gpu(
    # Perf (2026-07-13, task #163 stage 5 follow-up): ONE dispatch over
    # ALL n_pix*_BDPT_MAX_VERTS shadow-ray slots (replaces the earlier
    # _BDPT_MAX_VERTS-separate-dispatches loop) -- `shadow_results` is the
    # RAW interop trace output for the WHOLE batch, indexed by the same
    # flat `tid` as `shadow_pending`/`shadow_valid`/`shadow_seg_med`/
    # `shadow_rays` (all already strided idx=pix*_BDPT_MAX_VERTS+local, so
    # no re-indexing is needed -- `tid` IS `idx`). Requires the interop
    # scene's ray capacity to have been sized for n_pix*_BDPT_MAX_VERTS
    # (see pipeline.mojo's max_rays_vk_vcm), not just n_pix.
    #
    # Reads the RAW interop trace output directly (same idx*8 float layout
    # vulkaninterop_unpack_results_kernel consumes) instead of going
    # through a separate unpack-into-Intersection_C kernel first -- this
    # kernel only ever needs hit/material-index, not the full reconstructed
    # Intersection_C (uv/mesh/tri/area-light-index).
    #
    # `scratch` is a SEPARATE Intersection_C buffer, sized for
    # n_pix*_BDPT_MAX_VERTS (one slot PER THREAD, offset by `tid` below) for
    # _visible_transmittance's own internal multi-round tracing in the
    # fallback path -- must not alias `shadow_pending`/`shadow_valid`/
    # `shadow_seg_med`/`shadow_rays`.
    shadow_results: Pointer[Float32, MutUntrackedOrigin],
    mesh_material_idx: Pointer[Int64, MutUntrackedOrigin],
    n_meshes_vk_dp: Int64,
    cam_states: Pointer[VCMCameraPathState_C, MutUntrackedOrigin],
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin],
    shadow_rays: Pointer[Float32, MutUntrackedOrigin],
    scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    count_dp: Int64,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
):
    var spectral_res = Int(spectral_res_dp)
    var n_meshes_vk = Int(n_meshes_vk_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid
    if shadow_valid[unsafe_offset=idx] == Int8(0):
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

    var seg_med = shadow_seg_med[unsafe_offset=idx]
    var idx8 = idx * 8
    var org = Point3f(shadow_rays[unsafe_offset=idx8 + 0], shadow_rays[unsafe_offset=idx8 + 1], shadow_rays[unsafe_offset=idx8 + 2])
    var dir = Vec3f(shadow_rays[unsafe_offset=idx8 + 4], shadow_rays[unsafe_offset=idx8 + 5], shadow_rays[unsafe_offset=idx8 + 6])
    var dist = shadow_rays[unsafe_offset=idx8 + 7] / Float32(0.9995)

    var ridx = tid * 8
    var iresults = shadow_results.unsafe_bitcast[Int32]()
    var hitFlag = iresults[unsafe_offset=ridx + 6]

    var needs_fallback = False
    if hitFlag != Int32(1):
        if seg_med >= Int32(0):
            needs_fallback = True
        # else: fully visible (Tr=1), pending already holds the correct
        # unweighted contribution -- nothing to multiply.
    else:
        var mi = Int(iresults[unsafe_offset=ridx + 4])
        var mat_idx = Int64(0)
        if mi >= 0 and mi < n_meshes_vk:
            mat_idx = mesh_material_idx[unsafe_offset=mi]
        var mat = sd.materials[unsafe_offset=Int(mat_idx)]
        if seg_med >= Int32(0) or mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric or mat.type == MatKind.interface:
            needs_fallback = True
        else:
            # Opaque hit, no medium: fully occluded.
            shadow_pending[unsafe_offset=idx] = SpectralSample(Float32(0))

    if needs_fallback:
        # Per-thread scratch slot (offset by `tid`) -- matches the
        # established convention elsewhere in this file (e.g.
        # _bdpt_camera_connect_gpu's `scratch = inter_scratch + pix`).
        # Passing the bare pointer here would race across threads; harmless
        # today only because neither of this task's test scenes ever
        # actually enters this fallback path (no dielectric/medium), but
        # fixed now while this kernel is being rewritten anyway.
        var dst = org + dir * dist
        var cst = cam_states[unsafe_offset=idx // _BDPT_MAX_VERTS]
        var wl_sp = SampledWavelengths(cst.wl0, cst.wl1, cst.wl2, cst.wl3, cst.wl_pdf)
        var Tr = _visible_transmittance(org, dst, seg_med, sd, scratch.unsafe_offset(tid), wl_sp)
        var p = shadow_pending[unsafe_offset=idx]
        shadow_pending[unsafe_offset=idx] = p * Tr

def sum_shadow_connect_gpu(
    states: Pointer[VCMCameraPathState_C, MutUntrackedOrigin],
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    n_pix_dp: Int64,
):
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var base = pix * _BDPT_MAX_VERTS
    var sum = SpectralSample(Float32(0))
    for local in range(_BDPT_MAX_VERTS):
        if shadow_valid[unsafe_offset=base + local] != Int8(0):
            sum += shadow_pending[unsafe_offset=base + local]
    states[unsafe_offset=pix].total += sum

def vcm_render_gpu_wavefront(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
    n_spp:    Int,
    n_photons_req: Int,
    no_denoise: Bool,
    verbose:  Bool,
    # Task #163 stage 4 part 4: when set, both subpaths' per-bounce
    # intersect is routed through the CUDA/Vulkan interop mechanism
    # (vulkaninterop_rt_traverse_light_paths_gpu/_camera_) instead of
    # traverse_bvh2_core/test_spheres -- see that section's own comment.
    # interop_scene must be sized for max(n_light_paths_merge, n_pix) rays
    # (n_light_paths_merge is always >= n_pix by construction, see
    # n_photons_req's docstring below) since it's reused sequentially for
    # both the light pass and the camera pass every sample.
    use_vk: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
) -> Int32:
    """Task #163 stage 4 part 3: wavefront-staged variant of vcm_render_gpu,
    using _bdpt_light_path_init/_intersect/_bounce_gpu and
    _bdpt_camera_path_init/_intersect/_bounce_gpu instead of the single
    _bdpt_emit_light_paths_gpu/_bdpt_camera_connect_gpu mega-kernels --
    same algorithm, same LVC-then-connect ordering (light pass runs to
    full completion before the camera pass starts, since camera vertices
    connect/merge against the COMPLETE light-path cache, not a
    partially-built one), same output. The only behavioral difference is
    control-flow SHAPE: many small kernel launches (one intersect+bounce
    pair per _BDPT_MAX_DEPTH depth level, per subpath) instead of two
    single-kernel-traces-a-whole-subpath launches. Task #163 stage 4 part
    4: `use_vk=True` routes that per-bounce intersect through the CUDA/
    Vulkan interop mechanism (real hardware ray-query tracing, stage-1/2/3)
    instead of `traverse_bvh2_core`/`test_spheres` -- see this file's
    `vulkaninterop_rt_traverse_light_paths_gpu`/`_camera_` and their own
    section comment. `use_vk=False` (default) keeps the software-BVH path,
    still useful as a same-algorithm baseline for A/B comparison. Shadow
    rays (NEE/connect/merge/MNEE) stay on software BVH in BOTH modes, per
    the user-confirmed stage 4 scope -- only the primary/bounce ray is
    ever Vulkan-RT-traced. See vcm_render_gpu's own docstring for the
    shared algorithm-level documentation (n_photons_req/n_light_paths_merge
    derivation etc.), not repeated here."""
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)
    var max_comp  = psc[unsafe_offset=0].film_max_comp
    var n_light_paths_merge = max(n_photons_req, n_pix)

    print("VCM (GPU wavefront): " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
          + String(n_light_paths_merge) + " light paths/pass")

    var has_med = Int(sd.mediumCount) > 0
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[unsafe_offset=mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    var lvc_cap = n_light_paths_merge * _BDPT_MAX_VERTS
    var base_seed = psc[unsafe_offset=0].rng_seed

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
            var light_states_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths_merge, 1) * size_of[VCMLightPathState_C]())
            var cam_states_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[VCMCameraPathState_C]())
            # Task #163 stage 5: diffuse-branch connect shadow-ray queue,
            # strided _BDPT_MAX_VERTS slots per pixel -- see
            # _bdpt_connect_to_cache_deferred/resolve_shadow_connect_gpu.
            # Only allocated/used when use_vk (software-BVH _connect stays
            # the only path otherwise).
            var shadow_cap = n_pix * _BDPT_MAX_VERTS
            var shadow_rays_buf    = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * 8 * size_of[Float32]())
            var shadow_pending_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[SpectralSample]())
            var shadow_valid_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Int8]())
            var shadow_seg_med_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Int32]())
            # One Intersection_C scratch slot PER THREAD (not per pixel) for
            # resolve_shadow_connect_gpu's _visible_transmittance fallback
            # call -- inter_light_buf (sized n_light_paths_merge) is too
            # small now that resolve dispatches n_pix*_BDPT_MAX_VERTS
            # threads in one go (2026-07-13 perf follow-up).
            var shadow_scratch_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Intersection_C]())
            var accum_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)
            var albedo_accum_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with albedo_accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)

            var r2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with r2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].raster_to_camera.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            var c2w_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with c2w_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].camera_to_world.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            var lvc_ptr     = lvc_buf.unsafe_ptr().unsafe_bitcast[BDPTVertex]()
            var path_len_ptr = path_len_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_heads_ptr = merge_heads_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_next_ptr  = merge_next_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var inter_light_ptr = inter_light_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]()
            var inter_cam_ptr   = inter_cam_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]()
            var light_states_ptr = light_states_buf.unsafe_ptr().unsafe_bitcast[VCMLightPathState_C]()
            var cam_states_ptr   = cam_states_buf.unsafe_ptr().unsafe_bitcast[VCMCameraPathState_C]()
            var shadow_rays_ptr    = shadow_rays_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var shadow_pending_ptr = shadow_pending_buf.unsafe_ptr().unsafe_bitcast[SpectralSample]()
            var shadow_valid_ptr   = shadow_valid_buf.unsafe_ptr().unsafe_bitcast[Int8]()
            var shadow_seg_med_ptr = shadow_seg_med_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var shadow_scratch_ptr = shadow_scratch_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]()
            # Task #163 stage 5 perf fix #3 (2026-07-13): scene-adaptive
            # shadow-ray batching. Investigation (dragon vs cornell-box)
            # found the one-shot n_pix*_BDPT_MAX_VERTS dispatch (perf fix
            # #2, commit cb25f83) only pays off when light paths are long
            # enough to fill a real fraction of those _BDPT_MAX_VERTS
            # slots -- cornell-box's mean path_len is 5.18/10 (~52%
            # occupancy, a real win at every tested scale); dragon's is
            # 0.51/10 (~5% occupancy -- the dispatch is ~95% wasted
            # degenerate-ray padding, and got WORSE than software, even
            # worse than the earlier primary-ray-only swap). Decided ONCE
            # per render (not per sample -- a per-sample host sync was
            # already tried and found catastrophic, see perf fix #1's
            # revert note below) via a single path_len_buf readback after
            # si=0's light pass, before its camera pass starts. The 20%-
            # occupancy threshold is a judgment call from exactly 2 data
            # points (cornell-box/dragon), not a rigorously derived
            # constant -- revisit if a 3rd scene disagrees.
            var shadow_batch_enabled = use_vk
            var accum_ptr   = accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var albedo_accum_ptr = albedo_accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            # t=1 light tracing needs the same two camera-projection
            # matrices vcm_render / vcm_render_gpu build (see vcm_render's
            # comment): w2c = inverse(cameraToWorld), and c2r inverting the
            # 3x3 that turns (filmX, filmY, 1) into a camera-space direction.
            var w2c_host = unsafe_alloc[Float32](16)
            _ = matrix_invert(psc[unsafe_offset=0].camera_to_world, w2c_host)
            var _r2c_h = psc[unsafe_offset=0].raster_to_camera
            var a0 = _r2c_h[unsafe_offset=0]; var a1 = _r2c_h[unsafe_offset=4]; var a2 = _r2c_h[unsafe_offset=12]
            var b0 = _r2c_h[unsafe_offset=1]; var b1 = _r2c_h[unsafe_offset=5]; var b2 = _r2c_h[unsafe_offset=13]
            var g0 = _r2c_h[unsafe_offset=2]; var g1 = _r2c_h[unsafe_offset=6]; var g2 = _r2c_h[unsafe_offset=14]
            var d0 = b1*g2 - b2*g1
            var d1 = b0*g2 - b2*g0
            var d2 = b0*g1 - b1*g0
            var det = a0*d0 - a1*d1 + a2*d2
            var idet = Float32(1) / det if abs(det) > Float32(1e-20) else Float32(0)
            var c2r_host = unsafe_alloc[Float32](9)
            c2r_host[unsafe_offset=0] =  d0*idet; c2r_host[unsafe_offset=1] = -(a1*g2 - a2*g1)*idet; c2r_host[unsafe_offset=2] =  (a1*b2 - a2*b1)*idet
            c2r_host[unsafe_offset=3] = -d1*idet; c2r_host[unsafe_offset=4] =  (a0*g2 - a2*g0)*idet; c2r_host[unsafe_offset=5] = -(a0*b2 - a2*b0)*idet
            c2r_host[unsafe_offset=6] =  d2*idet; c2r_host[unsafe_offset=7] = -(a0*g1 - a1*g0)*idet; c2r_host[unsafe_offset=8] =  (a0*b1 - a1*b0)*idet
            var w2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with w2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = w2c_host.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            var c2r_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](9 * size_of[Float32]())
            with c2r_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = c2r_host.unsafe_bitcast[UInt8]()
                for i in range(9 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            w2c_host.unsafe_free()
            c2r_host.unsafe_free()
            var w2c_ptr = w2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2r_ptr = c2r_buf.unsafe_ptr().unsafe_bitcast[Float32]()

            var bvh2Nodes = handle[].bvh.nodes_ptr()
            var primIds = handle[].bvh.prim_ids_ptr()
            var meshes = handle[].meshes.meshes_ptr()
            var curves = handle[].curves.curves_ptr()
            var blasNodesArr = handle[].blas.nodes_arr()
            var blasPrimIdsArr = handle[].blas.primids_arr()
            var instances = handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C]()
            var materials = handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C]()
            var mediums = handle[].mediums_buf.unsafe_ptr().unsafe_bitcast[Medium_C]()
            # Device-resident density fields for the free-flight sampler. These
            # were never handed to the VCM/SPPM kernels before, which is exactly
            # why those integrators sampled every heterogeneous medium as uniform
            # density-1 fog -- see geometry.mojo's sample_free_flight.
            var grids_dev = handle[].grids_buf.unsafe_ptr().unsafe_bitcast[Grid_C]()
            var nvdb_grids_dev = handle[].nvdb_grids_buf.unsafe_ptr().unsafe_bitcast[NvdbGrid_C]()
            var mediumInterfaces = handle[].medium_ifaces_buf.unsafe_ptr().unsafe_bitcast[MediumInterface_C]()
            var spheres = handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C]()
            var areaLights = handle[].lights.area_lights_ptr()
            var distantLights = handle[].lights.distant_lights_ptr()
            var infiniteLights = handle[].lights.infinite_lights_ptr()
            var pointLights = handle[].lights.point_lights_ptr()
            var n_mediums = Int64(handle[].n_mediums)
            var n_medium_ifaces = Int64(handle[].n_medium_ifaces)
            var n_spheres = Int64(handle[].n_spheres)
            var n_curves = Int64(handle[].curves.n_curves)
            var n_area_lights = Int64(handle[].lights.n_area_lights)
            var n_distant_lights = Int64(handle[].lights.n_distant_lights)
            var n_infinite_lights = Int64(handle[].lights.n_infinite_lights)
            var n_point_lights = Int64(handle[].lights.n_point_lights)
            var n_blas = Int64(handle[].blas.n_blas)
            var n_instances = Int64(handle[].n_instances)
            var (spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) = handle[].spectral.unsafe_ptrs()
            var measured_brdfs = handle[].measured_brdfs_buf.unsafe_ptr().unsafe_bitcast[MeasuredBRDF_C]()
            var n_measured_brdfs = Int64(handle[].n_measured_brdfs)
            var gpu_textures = handle[].textures.textures_ptr()
            var n_gpu_textures = Int64(handle[].textures.n_textures)

            var grid_light = ceildiv(max(n_light_paths_merge, 1), block_size)
            var grid_pix = ceildiv(n_pix, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            var (_scene_center, scene_radius) = _scene_bounding_sphere(sd)
            var px_scale = Float32(2.0) * tan(psc[unsafe_offset=0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(fh)
            var n_light_paths_f = Float32(n_light_paths_merge)

            var grid_merge_ins = ceildiv(max(lvc_cap, 1), block_size)

            for si in range(n_spp):
                # Stage 2c progressive radius -- see vcm_render (CPU)'s
                # matching per-sample loop for the full derivation comment.
                var radius_i = vcm_merge_radius(scene_radius, si)
                var merge_r2 = radius_i * radius_i
                var merge_inv_cell = Float32(1.0) / max(radius_i, Float32(1e-6))
                var merge_norm = Float32(1.0) / (Float32(n_light_paths_merge) * PI * max(merge_r2, Float32(1e-12)))
                var eta_vcm = PI * max(merge_r2, Float32(1e-12)) * Float32(n_light_paths_merge)
                var mis_vm_weight_factor = eta_vcm
                var mis_vc_weight_factor = Float32(1.0) / eta_vcm

                var pass_seed = base_seed ^ UInt64(si * 2654435761 + 1)

                # Task #163 stage 4 part 3: wavefront-staged light pass --
                # init once, then intersect+bounce once per depth level,
                # instead of _bdpt_emit_light_paths_gpu's single mega-kernel.
                # Every kernel internally skips lanes whose state has already
                # gone inactive (mirrors gpu.mojo's traverse_paths_gpu/
                # shade_*_gpu `if paths[tid].active == 0: return` convention)
                # -- running the full _BDPT_MAX_DEPTH iterations regardless
                # of how many lanes are still active is the same fixed-
                # iteration-count wavefront shape the plain path tracer uses.
                handle[].ctx.enqueue_function[_bdpt_light_path_init_gpu](
                    light_states_ptr, lvc_ptr, path_len_ptr, mis_vc_weight_factor,
                    Int64(n_light_paths_merge), default_emit_med, pass_seed, Int64(si),
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grid_dim=grid_light, block_dim=block_size)

                for _bounce_i in range(_BDPT_MAX_DEPTH):
                    if use_vk:
                        vulkaninterop_rt_traverse_light_paths_gpu(
                            handle[].ctx, light_states_buf, inter_light_buf, interop_scene,
                            interop_rays_buf.value(), interop_results_buf.value(),
                            mesh_material_idx_buf.value(), mesh_al_idx_buf.value(),
                            n_meshes_vk, n_light_paths_merge)
                    else:
                        handle[].ctx.enqueue_function[_bdpt_light_path_intersect_gpu](
                            bvh2Nodes, primIds, meshes, curves,
                            blasNodesArr, blasPrimIdsArr, instances,
                            spheres, Int64(Int(n_spheres)),
                            light_states_ptr, inter_light_ptr, Int64(n_light_paths_merge),
                            grid_dim=grid_light, block_dim=block_size)
                    handle[].ctx.enqueue_function[_bdpt_light_path_bounce_gpu](
                        light_states_ptr, inter_light_ptr, lvc_ptr, path_len_ptr,
                        mis_vc_weight_factor, mis_vm_weight_factor, Int64(n_light_paths_merge),
                        c2w_ptr, px_scale,
                        bvh2Nodes, primIds, meshes, materials,
                        areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                        mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                        blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                        distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                        pointLights, n_point_lights,
                        spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                        measured_brdfs, n_measured_brdfs,
                        gpu_textures, n_gpu_textures,
                        grids_dev, Int64(handle[].n_grids), nvdb_grids_dev, Int64(handle[].n_nvdb_grids),
                        grid_dim=grid_light, block_dim=block_size)

                # VCM Stage 2b: light paths are deterministically paired with
                # pixels (the first n_pix of n_light_paths_merge total light
                # paths, task #152), so no host readback of a total vertex
                # count is needed anymore -- lvc_cap is already known at
                # compile/host time. Kernels stay ordered on one stream
                # without an explicit synchronize() here.
                handle[].ctx.enqueue_function[bdpt_merge_grid_reset_gpu](
                    merge_heads_ptr, Int64(_HSIZE), grid_dim=grid_hsize, block_dim=block_size)
                handle[].ctx.enqueue_function[bdpt_merge_grid_insert_gpu](
                    lvc_ptr, path_len_ptr, Int64(lvc_cap), merge_next_ptr, merge_heads_ptr, merge_inv_cell,
                    grid_dim=grid_merge_ins, block_dim=block_size)

                if use_vk and si == 0:
                    var sum_pl = 0
                    with path_len_buf.map_to_host() as host_buf:
                        var pl = host_buf.unsafe_ptr().unsafe_bitcast[Int32]()
                        for pli in range(n_light_paths_merge):
                            sum_pl += Int(pl[unsafe_offset=pli])
                    var occupancy = (Float64(sum_pl) / Float64(n_light_paths_merge)) / Float64(_BDPT_MAX_VERTS)
                    if occupancy < 0.2:
                        shadow_batch_enabled = False
                    if verbose:
                        print("VCM (GPU wavefront): shadow-ray batching " + ("enabled" if shadow_batch_enabled else "disabled")
                              + " (light-path slot occupancy " + String(occupancy * 100) + "%)")

                # Task #163 stage 5 perf fix, ATTEMPTED AND REVERTED
                # (2026-07-13): tried bounding the per-bounce shadow-ray
                # local-slot loop below by this sample's actual max light-
                # path length (one path_len_buf.map_to_host() readback per
                # SAMPLE). Correct in principle, but made things WORSE, not
                # better -- 256x256/16spp went from 6.4s to 34.9s. Root
                # cause: this whole per-sample loop is normally fully async/
                # pipelined across all `si` iterations with zero host syncs;
                # a single per-sample sync point breaks that overlap and
                # forces the CPU to idle-wait on GPU completion every
                # sample, which cost far more than the dispatches it saved.
                # Also, VCM's bounce loop doesn't respect the scene's own
                # `maxdepth` (grep confirms bdpt.mojo never reads
                # psc[0].max_depth) -- cornell-box's light paths routinely
                # reach the full _BDPT_MAX_VERTS(10) cap via RR-only
                # termination regardless of its maxdepth=4 setting, so even
                # a static maxdepth-based bound would be unsafe (could drop
                # real connect candidates) as well as ineffective (wouldn't
                # actually reduce shadow_locals below 10 for this scene).
                # See project_vulkan_rt_backend memory for the full story.

                # Task #163 stage 4 part 3: wavefront-staged camera pass --
                # same shape as the light pass above. Connect/merge (still
                # software-BVH shadow rays, per the user-confirmed stage 4
                # scope) run inline inside _bdpt_camera_path_bounce_gpu at
                # every non-delta vertex, exactly where
                # _bdpt_camera_connect_gpu's single mega-kernel already ran
                # them -- only the primary/bounce ray intersect moved out.
                handle[].ctx.enqueue_function[_bdpt_camera_path_init_gpu](
                    cam_states_ptr, r2c_ptr, c2w_ptr, Int64(n_pix), Int64(Int(psc[unsafe_offset=0].film_w)),
                    px_scale, n_light_paths_f,
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    base_seed, Int64(si),
                    grid_dim=grid_pix, block_dim=block_size)

                for _bounce_i in range(_BDPT_MAX_DEPTH):
                    if use_vk:
                        vulkaninterop_rt_traverse_camera_paths_gpu(
                            handle[].ctx, cam_states_buf, inter_cam_buf, interop_scene,
                            interop_rays_buf.value(), interop_results_buf.value(),
                            mesh_material_idx_buf.value(), mesh_al_idx_buf.value(),
                            n_meshes_vk, n_pix)
                    else:
                        handle[].ctx.enqueue_function[_bdpt_camera_path_intersect_gpu](
                            bvh2Nodes, primIds, meshes, curves,
                            blasNodesArr, blasPrimIdsArr, instances,
                            spheres, Int64(Int(n_spheres)),
                            cam_states_ptr, inter_cam_ptr, Int64(n_pix),
                            grid_dim=grid_pix, block_dim=block_size)
                    if shadow_batch_enabled:
                        handle[].ctx.enqueue_function[reset_shadow_valid_gpu](
                            shadow_valid_ptr, Int64(n_pix), grid_dim=grid_pix, block_dim=block_size)
                    handle[].ctx.enqueue_function[_bdpt_camera_path_bounce_gpu](
                        cam_states_ptr, inter_cam_ptr, lvc_ptr, path_len_ptr,
                        merge_next_ptr, merge_heads_ptr, merge_inv_cell, merge_r2, merge_norm,
                        mis_vc_weight_factor, mis_vm_weight_factor, Int64(n_pix),
                        c2w_ptr, px_scale,
                        bvh2Nodes, primIds, meshes, materials,
                        areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                        mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                        blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                        distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                        pointLights, n_point_lights,
                        spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                        measured_brdfs, n_measured_brdfs,
                        gpu_textures, n_gpu_textures,
                        Int8(1) if shadow_batch_enabled else Int8(0), shadow_rays_ptr, shadow_pending_ptr, shadow_valid_ptr, shadow_seg_med_ptr,
                        grids_dev, Int64(handle[].n_grids), nvdb_grids_dev, Int64(handle[].n_nvdb_grids),
                        grid_dim=grid_pix, block_dim=block_size)

                    # Task #163 stage 5 perf follow-up (2026-07-13): resolve
                    # this bounce's diffuse-branch connect shadow rays in
                    # ONE dispatch over all n_pix*_BDPT_MAX_VERTS slots
                    # (the interop scene's ray capacity was resized in
                    # pipeline.mojo specifically for this), replacing the
                    # earlier _BDPT_MAX_VERTS-separate-dispatches loop --
                    # cuts the per-bounce interop trace CALL count from 10
                    # to 1, the dominant cost identified by that follow-up's
                    # investigation. Then fold the resolved contributions
                    # into each pixel's running total.
                    if shadow_batch_enabled:
                        var shadow_grid = ceildiv(shadow_cap, block_size)
                        vulkaninterop_rt_traverse_shadow_gpu(
                            handle[].ctx, shadow_rays_buf, shadow_valid_buf,
                            interop_scene, interop_rays_buf.value(),
                            shadow_cap)
                        handle[].ctx.enqueue_function[resolve_shadow_connect_gpu](
                            interop_results_buf.value().unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                            mesh_material_idx_buf.value().unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                            Int64(n_meshes_vk),
                            cam_states_ptr, shadow_pending_ptr, shadow_valid_ptr, shadow_seg_med_ptr, shadow_rays_ptr,
                            shadow_scratch_ptr, Int64(shadow_cap),
                            bvh2Nodes, primIds, meshes, materials,
                            areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                            mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                            blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                            distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                            pointLights, n_point_lights,
                            spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                            measured_brdfs, n_measured_brdfs,
                            gpu_textures, n_gpu_textures,
                            grid_dim=shadow_grid, block_dim=block_size)
                        handle[].ctx.enqueue_function[sum_shadow_connect_gpu](
                            cam_states_ptr, shadow_pending_ptr, shadow_valid_ptr, Int64(n_pix),
                            grid_dim=grid_pix, block_dim=block_size)

                handle[].ctx.enqueue_function[_bdpt_camera_path_accumulate_gpu](
                    cam_states_ptr, accum_ptr, albedo_accum_ptr, Int64(n_pix),
                    sd.spectral.coeffs, Int64(sd.spectral.res), sd.spectral.cie_x,
                    sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                    grid_dim=grid_pix, block_dim=block_size)

                # Phase 1.5: t=1 light tracing -- the SAME kernel the
                # megakernel driver launches, reading the same fully-built
                # LVC. Launched after the camera accumulate on the same
                # stream so that kernel's non-atomic per-pixel writes can
                # never overlap these atomic adds.
                handle[].ctx.enqueue_function[_bdpt_splat_light_paths_gpu](
                    accum_ptr, lvc_ptr, path_len_ptr, Int64(n_light_paths_merge),
                    inter_light_ptr, w2c_ptr, c2r_ptr, c2w_ptr,
                    Int64(fw), Int64(fh), px_scale, mis_vm_weight_factor,
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grid_dim=grid_light, block_dim=block_size)

                if verbose:
                    print("VCM (GPU wavefront): sample " + String(si + 1) + "/" + String(n_spp))

            handle[].ctx.synchronize()

            var pixels = unsafe_alloc[Float32](n_pix * 3)
            with accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                var inv_spp = iso_scale / Float32(n_spp)
                for i in range(n_pix):
                    var r = src[unsafe_offset=i*3]   * inv_spp
                    var g = src[unsafe_offset=i*3+1] * inv_spp
                    var b = src[unsafe_offset=i*3+2] * inv_spp
                    if max_comp > Float32(0):
                        r = r if r < max_comp else max_comp
                        g = g if g < max_comp else max_comp
                        b = b if b < max_comp else max_comp
                    pixels[unsafe_offset=i*3] = r; pixels[unsafe_offset=i*3+1] = g; pixels[unsafe_offset=i*3+2] = b

            # Denoise (never wired up before -- no_denoise was a dead
            # parameter): read back the albedo AOV accumulated above, run
            # a fresh normals/depth pass via the host-side sd (same
            # render_aux_buffers the CPU path/plain tracer use -- host-only,
            # so it runs on the CPU here too, not as a GPU kernel), then the
            # same CPU denoise() the CPU BDPT path uses.
            var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
            with albedo_accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                var inv_spp_alb = Float32(1) / Float32(n_spp)
                for i in range(n_pix * 3):
                    albedo_pixels[unsafe_offset=i] = src[unsafe_offset=i] * inv_spp_alb

            var normals = unsafe_alloc[Float32](n_pix * 3)
            var depth = unsafe_alloc[Float32](n_pix)
            var sd_local = sd
            render_aux_buffers(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world, Int32(0), Int32(0),
                                psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h, Pointer(to=sd_local), normals, depth)

            var denoised = unsafe_alloc[Float32](n_pix * 3)
            if no_denoise:
                for i in range(n_pix * 3): denoised[unsafe_offset=i] = pixels[unsafe_offset=i]
            else:
                denoise(pixels, albedo_pixels, normals, depth, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
                        denoised, Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

            _ = write_image_cropwindow(denoised, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
        psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
        psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))
            pixels.unsafe_free(); albedo_pixels.unsafe_free(); normals.unsafe_free(); depth.unsafe_free(); denoised.unsafe_free()
        except e:
            print("VCM GPU wavefront render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("VCM GPU wavefront: no accelerator")
        ret = Int32(-1)
    return ret




# ── SPPM GPU driver + kernels (relocated from sppm.mojo) ─────────────────────
# Moved here as a workaround for a reproducible Mojo 1.0.0 / Modular 26.5.0
# compiler bug: every ctx.enqueue_function[kernel] call defined/used from
# WITHIN sppm.mojo (regardless of kernel/host-function shape -- verified down
# to a 3-line minimal repro, and independently for a totally fresh,
# unrelated kernel added to that file) fails to typecheck ("no matching
# method in call to 'enqueue_function'", the kernel's own inferred type
# showing as plain "thin -> None" instead of the expected "capturing
# thin -> None"). The identical code compiles cleanly once moved to
# bdpt.mojo (confirmed empirically, twice: once for an existing sppm.mojo
# kernel called cross-file, once for a brand-new kernel+host-function pair
# defined directly in bdpt.mojo) -- this appears to be a real, per-FILE-
# specific compiler defect (a new file created for this code, e.g.
# `sppm_gpu.mojo`, was ALSO cursed). See reference_mojo_compiler_bug_6759
# memory / project_modular_26_5_0_migration memory for the full bisection.
#
# Re-checked 2026-09-17 on the current toolchain, and the move still fails --
# but the recorded "it is specifically the sppm.mojo file object" reading is
# WRONG: a minimal kernel + enqueue_function pair appended to sppm.mojo
# compiles fine, while THESE kernels fail both there and in a fresh
# sppm_gpu.mojo, with explicit imports as well as a wildcard one. So the
# trigger travels with this code, not with a file. Whatever it is has not
# been isolated; a smaller repro than "move all 735 lines" is the next step
# for anyone retrying.
# ── GPU kernels ───────────────────────────────────────────────────────────────
# Each kernel is a thin wrapper: compute this thread's index, build a complete
# SceneDescriptor2_C via _mk_sd_full (bvh.mojo), then call the EXACT SAME
# shared function the CPU driver above calls (comptime[use_gpu]-branching
# only at the two genuine concurrency-primitive divergence points: photon-
# slot reservation and hash-grid bucket insertion) — mirrors bdpt.mojo's
# GPU kernels, deliberately unlike the old gpu_sppm.mojo (a full duplicate
# reimplementation).

def sppm_reset_i32_gpu(counter: Pointer[Int32, MutUntrackedOrigin]):
    if block_idx.x == 0 and thread_idx.x == 0:
        counter[unsafe_offset=0] = Int32(0)


def sppm_gen_vp_gpu(
    vps: Pointer[SPPMPixel, MutUntrackedOrigin],
    inter_scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    n_pix_dp: Int64,
    vp_samples_dp: Int64,
    fw: Int32,
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    init_r2: Float32,
    seed: UInt64,
    max_depth_dp: Int64,
    film_filter: FilmFilter,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
    # Device-resident density fields, for the free-flight sampler. Without
    # these the descriptor built below reports no density fields and every
    # heterogeneous medium samples as uniform density-1 fog.
    grids: Pointer[Grid_C, MutUntrackedOrigin] = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_grids: Int64 = Int64(0),
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin] = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_nvdb_grids: Int64 = Int64(0),
):
    """One thread per (pixel, vp_sample). Calls the SAME
    _sppm_trace_visible_point the CPU driver (_sppm_camera_pass) calls,
    with use_gpu=True (task #151: real image-texture reflectance)."""
    var n_pix = Int(n_pix_dp)
    var vp_samples = Int(vp_samples_dp)
    var combined = Int(block_idx.x * block_dim.x + thread_idx.x)
    if combined >= n_pix * vp_samples:
        return
    var pix = combined // vp_samples
    var px = pix % Int(fw)
    var py = pix // Int(fw)
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
        distantLights, distantLightCount, infiniteLights, infiniteLightCount,
        gpuTextures=gpuTextures, gpuTextureCount=gpuTextureCount,
        grids=grids, gridCount=n_grids, nvdbGrids=nvdb_grids, nvdbGridCount=n_nvdb_grids,
    )
    var pcg = PCG32(seed ^ UInt64(combined * 6364136223846793005 + 1), UInt64(1))
    vps[unsafe_offset=combined] = _sppm_trace_visible_point[True](sd, pcg, r2c, c2w, px, py, Int32(pix), init_r2, inter_scratch.unsafe_offset(combined), Int(max_depth_dp), film_filter)


def sppm_emit_photons_gpu(
    photons: Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_emit_dp: Int64,
    max_photons_dp: Int64,
    inter_scratch: Pointer[Intersection_C, MutUntrackedOrigin],
    stored_counter: Pointer[Int32, MutUntrackedOrigin],
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx_dp: Int64,
    max_depth_dp: Int64,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    # camera_to_world + pixel angular size: a photon needs a bump/normal-map
    # footprint and has no ray cone, so it uses the camera-distance
    # approximation -- see _sppm_trace_photon's own params.
    c2w: Pointer[Float32, MutUntrackedOrigin],
    photon_px_scale: Float32,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: Pointer[GpuTexture_C, MutUntrackedOrigin] = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
    # Device-resident density fields, for the free-flight sampler. Without
    # these the descriptor built below reports no density fields and every
    # heterogeneous medium samples as uniform density-1 fog.
    grids: Pointer[Grid_C, MutUntrackedOrigin] = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_grids: Int64 = Int64(0),
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin] = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(),
    n_nvdb_grids: Int64 = Int64(0),
):
    """One thread per emitted photon path. Calls the SAME _sppm_trace_photon
    the CPU driver (_sppm_photon_pass) calls, with use_gpu=True so
    _sppm_store_photon reserves its slot via an atomic fetch-add (CPU uses a
    plain counter increment instead — no other difference), and tex_gpu=True
    since this kernel genuinely runs on the GPU (task #151 -- see
    _sppm_trace_photon's docstring for why these are separate params). The
    spectral/measured params are new (measured BxDF support, see
    project_measured_bxdf memory): _sppm_trace_photon's measured branch
    calls bxdf_sample_measured, which needs both a valid spectral table
    (for the RGB conversion) and the measured-BRDF array -- unlike this
    kernel's other materials, which only draw a wavelength via PCG and
    never dereference sd.spectral."""
    var spectral_res = Int(spectral_res_dp)
    var n_emit = Int(n_emit_dp)
    var max_photons = Int(max_photons_dp)
    var pass_idx = Int(pass_idx_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    # sphereCount is part of the test because an analytic sphere can BE the
    # scene's only light (Sphere_C.isAreaLight); leaving it out made this
    # kernel return before emitting a single photon there, so SPPM fell back
    # to visible-point NEE alone. _sppm_trace_photon does the exact
    # "is any sphere emitting" scan and returns on its own if none is.
    if k >= n_emit or (areaLightCount == Int64(0) and distantLightCount == Int64(0)
                       and infiniteLightCount == Int64(0) and pointLightCount == Int64(0)
                       and sphereCount == Int64(0)):
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
        grids=grids, gridCount=n_grids, nvdbGrids=nvdb_grids, nvdbGridCount=n_nvdb_grids,
    )
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    _sppm_trace_photon[True, True](sd, pcg, inter_scratch.unsafe_offset(k), n_emit, photons, max_photons, stored_counter, default_emit_med, Int(max_depth_dp),
        _sppm_cam_pos(c2w), photon_px_scale,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, pass_wavelengths(pass_idx))


def sppm_grid_reset_gpu(heads: Pointer[Int32, MutUntrackedOrigin], hsize_dp: Int64):
    var hsize = Int(hsize_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= hsize:
        return
    _sppm_reset_grid_cell(heads, tid)


def sppm_grid_insert_gpu(
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_stored_dp: Int64,
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    var n_stored = Int(n_stored_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_stored:
        return
    _sppm_insert_photon[True](k, photons, heads, inv_cell)


def sppm_gather_gpu(
    vps:      Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix_dp:    Int64,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    pass_idx_dp: Int64 = Int64(0),
    med_arr_dp: Pointer[Medium_C, MutUntrackedOrigin] = Pointer[Medium_C, MutUntrackedOrigin].unsafe_dangling(),
    med_count_dp: Int64 = Int64(0),
    grids_dp: Pointer[Grid_C, MutUntrackedOrigin] = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(),
    nvdb_grids_dp: Pointer[NvdbGrid_C, MutUntrackedOrigin] = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(),
):
    """One thread per visible point. `sd` here only needs to be complete
    enough for _sppm_gather_one's hair branch (sd.materials/sd.curves) —
    plus, since measured BxDF support was added (see project_measured_bxdf
    memory), its mat_kind=3 branch too (sd.measuredBrdfs + sd.spectral,
    via _sppm_vp_brdf). Other SceneDescriptor2_C fields it builds are still
    unused by gather, so stay zeroed/dangling exactly like sppm_nee_gpu's
    own _mk_sd_full call."""
    var spectral_res = Int(spectral_res_dp)
    var n_pix = Int(n_pix_dp)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        Pointer[AreaLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0), curves, curveCount,
        Pointer[Medium_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[MediumInterface_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        instances, instanceCount,
        Pointer[DistantLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[InfiniteLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[PointLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        measuredBrdfs, measuredBrdfCount,
    )
    _sppm_gather_one(vps, i, photons, heads, inv_cell, sd, med_arr_dp, Int(med_count_dp),
                     grids_dp, nvdb_grids_dp,
                     spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
                     spectral_cie_z, spectral_d65, pass_wavelengths(Int(pass_idx_dp)))


def sppm_nee_gpu(
    vps:    Pointer[SPPMPixel, MutUntrackedOrigin],
    n_vps_dp:  Int64,
    seed:   UInt64,
    pass_idx_dp: Int64,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    mediumInterfaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin] = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
):
    """One thread per visible point. Calls the SAME _sppm_nee_one the CPU
    driver (_sppm_nee_update) calls. The only one of SPPM's 4 GPU kernels
    that needs the spectral device buffers (staged spectral rendering
    rollout, Stage 4 -- see project_spectral_rendering memory): VP
    generation and the photon pass sample their own wavelengths via PCG
    only (no table lookup needed to draw a wavelength), and the gather pass
    stays RGB by design (see _sppm_gather_one's docstring) -- only this
    NEE pass's direct-lighting term actually dereferences sd.spectral."""
    var spectral_res = Int(spectral_res_dp)
    var n_vps = Int(n_vps_dp)
    var pass_idx = Int(pass_idx_dp)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_vps:
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
    )
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + i), UInt64(11))
    _sppm_nee_one(vps, i, sd, pcg)


def sppm_finalize_gpu(
    vps:        Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix_dp:      Int64,
    vp_samples_dp: Int64,
    n_passes:   Int32,
    iso_scale:  Float32,
    max_comp:   Float32,
    out_pixels: Pointer[Float32, MutUntrackedOrigin],
    albedo_out: Pointer[Float32, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin],
    spectral_res_dp: Int64,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
):
    """One thread per pixel. Calls the SAME _sppm_finalize_one_pixel the CPU
    driver (sppm_render's tail loop) calls, plus the matching albedo AOV
    average for the denoiser (staged along with the rest of Stage 4-adjacent
    denoiser wiring — see project_spectral_rendering memory)."""
    var n_pix = Int(n_pix_dp)
    var vp_samples = Int(vp_samples_dp)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    var acc = _sppm_finalize_one_pixel(vps, i, vp_samples, n_passes, iso_scale, max_comp,
                                       spectral_coeffs, Int(spectral_res_dp), spectral_cie_x,
                                       spectral_cie_y, spectral_cie_z, spectral_d65)
    out_pixels[unsafe_offset=i * 3 + 0] = acc.r
    out_pixels[unsafe_offset=i * 3 + 1] = acc.g
    out_pixels[unsafe_offset=i * 3 + 2] = acc.b
    var alb = _sppm_finalize_albedo_one_pixel(vps, i, vp_samples)
    albedo_out[unsafe_offset=i * 3 + 0] = alb.r
    albedo_out[unsafe_offset=i * 3 + 1] = alb.g
    albedo_out[unsafe_offset=i * 3 + 2] = alb.b


# ── GPU host driver ───────────────────────────────────────────────────────────

def sppm_render_gpu(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """GPU-accelerated Stochastic Progressive Photon Mapping — same algorithm
    as sppm_render, parallelized: one thread per visible-point sample for the
    camera pass/gather/NEE/finalize, one thread per emitted photon for the
    photon pass, atomic-exchange hash-grid build (classic parallel linked-list
    insertion). Mirrors vcm_render_gpu's per-pass reset-counter -> emit ->
    sync+readback+clamp -> consume shape."""
    if Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount) == 0 and not _sppm_has_sphere_lights(sd):
        print("SPPM: no lights in scene, cannot emit photons")
        return Int32(-1)

    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)
    var max_comp = psc[unsafe_offset=0].film_max_comp

    print("SPPM (GPU): " + String(fw) + "x" + String(fh)
          + " " + String(n_passes) + " passes x "
          + String(n_photons_per_pass) + " photons  r=" + String(initial_radius))

    var default_emit_med = Int32(-1)
    if Int(sd.mediumCount) > 0 and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[unsafe_offset=mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # A BSSRDF visible point gathers out to the material's DIFFUSION reach,
    # which for skin is several times the SPPM radius this scene would
    # otherwise pick. The photon grid's cells are initial_radius-sized and the
    # gather looks at 3x3x3 of them, so the radius has to cover that reach or
    # the neighbour search silently misses the photons carrying the subsurface
    # signal. Widening it costs nothing on this path: R_d is normalised, so
    # unlike a density estimate the radius does not scale the answer.
    var eff_radius = initial_radius
    for _mi in range(Int(sd.mediumCount)):
        if sd.mediums[unsafe_offset=_mi].is_sss != Int32(0):
            var _rq = dipole_max_radius(sd.mediums[unsafe_offset=_mi].sigma_s, sd.mediums[unsafe_offset=_mi].sigma_a, sd.mediums[unsafe_offset=_mi].g)
            if _rq > eff_radius:
                eff_radius = _rq
    # init_r2 stays on the SCENE's radius -- widening it would blur every
    # ordinary surface visible point in the scene. Only the grid CELLS grow,
    # so the 3x3x3 neighbour search can reach the diffusion distance; a larger
    # cell never changes a surface gather, it only searches more candidates.
    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / eff_radius
    if verbose:
        print("SPPM: vp radius " + String(initial_radius) + ", grid cell " + String(eff_radius))

    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256

            var n_vps = n_pix * _VP_SAMPLES
            # Sized for the worst case, mirroring sppm.mojo's CPU driver
            # (_sppm_render_core) exactly -- see its comment for why
            # max_photons must scale with the per-photon bounce budget, not
            # just n_photons_per_pass (one emitted path can store up to
            # min(maxdepth, _MAX_B) - 1 deposits, not one).
            var max_bounces_per_photon = min(Int(psc[unsafe_offset=0].max_depth), _MAX_B)
            # A subsurface interior blows this budget wide open: its random-walk
            # steps are deliberately NOT charged to maxdepth (see
            # _sppm_trace_photon's loop header), so one photon entering skin
            # deposits at every scatter for as long as the walk survives --
            # hundreds of events, not `maxdepth` of them. Sized for maxdepth
            # alone, _sppm_store_photon's shared atomic counter saturates almost
            # immediately (head.pbrt: stored hit exactly n_photons*maxdepth on
            # every pass) and the estimator still divides by the full emitted
            # count, leaving the surviving deposits' local density wildly
            # inflated. Mirrors the same sizing in sppm.mojo's CPU driver.
            var has_sss_medium = False
            for mi in range(Int(sd.mediumCount)):
                if sd.mediums[unsafe_offset=mi].is_sss != Int32(0):
                    has_sss_medium = True
                    break
            # (The "--sppm + subsurface is unsupported" warning that stood here
            # was WRONG and has been removed. It rested on one experiment --
            # 16x photons changing nothing -- which showed the estimate was
            # BIASED, not undersampled, i.e. a bug rather than a limit. With
            # subsurface transport evaluated by a surface-side diffusion BSSRDF
            # (bssrdf.mojo) instead of by photon-mapping the interior, head
            # renders at 0.86x the reference with no black pixels.)
            if has_sss_medium:
                max_bounces_per_photon += SSS_WALK_ROUNDS
            var max_photons = n_photons_per_pass * max(max_bounces_per_photon, 1)
            var vps_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](n_vps * size_of[SPPMPixel]())
            var photons_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(max_photons, 1) * size_of[SPPMPhoton]())
            var heads_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](_HSIZE * size_of[Int32]())
            var inter_cam_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_vps * size_of[Intersection_C]())
            var inter_ph_buf  = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_photons_per_pass, 1) * size_of[Intersection_C]())
            var counter_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](size_of[Int32]())
            var out_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            var albedo_out_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())

            var r2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with r2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].raster_to_camera.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            var c2w_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with c2w_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].camera_to_world.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            var vps_ptr    = vps_buf.unsafe_ptr().unsafe_bitcast[SPPMPixel]().unsafe_origin_cast[MutUntrackedOrigin]()
            var photons_ptr = photons_buf.unsafe_ptr().unsafe_bitcast[SPPMPhoton]().unsafe_origin_cast[MutUntrackedOrigin]()
            var heads_ptr  = heads_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var inter_cam_ptr = inter_cam_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_origin_cast[MutUntrackedOrigin]()
            var inter_ph_ptr  = inter_ph_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_origin_cast[MutUntrackedOrigin]()
            var counter_ptr = counter_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var out_ptr     = out_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var albedo_out_ptr = albedo_out_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            # Bump/normal-map footprint for the photon pass, computed HOST-side
            # from the same matrices the visible-point pass uses -- see
            # sppm.mojo's _sppm_photon_px_scale. The kernel gets the device
            # c2w (for the camera position) plus this one scalar.
            var photon_px_scale = _sppm_photon_px_scale(
                psc[unsafe_offset=0].raster_to_camera,
                psc[unsafe_offset=0].camera_to_world,
                Int(psc[unsafe_offset=0].film_w), Int(psc[unsafe_offset=0].film_h))

            var bvh2Nodes = handle[].bvh.nodes_ptr()
            var primIds = handle[].bvh.prim_ids_ptr()
            var meshes = handle[].meshes.meshes_ptr()
            var curves = handle[].curves.curves_ptr()
            var blasNodesArr = handle[].blas.nodes_arr()
            var blasPrimIdsArr = handle[].blas.primids_arr()
            var instances = handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C]()
            var materials = handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C]()
            var mediums = handle[].mediums_buf.unsafe_ptr().unsafe_bitcast[Medium_C]()
            # Device-resident density fields for the free-flight sampler. These
            # were never handed to the VCM/SPPM kernels before, which is exactly
            # why those integrators sampled every heterogeneous medium as uniform
            # density-1 fog -- see geometry.mojo's sample_free_flight.
            var grids_dev = handle[].grids_buf.unsafe_ptr().unsafe_bitcast[Grid_C]()
            var nvdb_grids_dev = handle[].nvdb_grids_buf.unsafe_ptr().unsafe_bitcast[NvdbGrid_C]()
            var mediumInterfaces = handle[].medium_ifaces_buf.unsafe_ptr().unsafe_bitcast[MediumInterface_C]()
            var spheres = handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C]()
            var areaLights = handle[].lights.area_lights_ptr()
            var distantLights = handle[].lights.distant_lights_ptr()
            var infiniteLights = handle[].lights.infinite_lights_ptr()
            var pointLights = handle[].lights.point_lights_ptr()
            var n_mediums = Int64(handle[].n_mediums)
            var n_medium_ifaces = Int64(handle[].n_medium_ifaces)
            var n_spheres = Int64(handle[].n_spheres)
            var n_curves = Int64(handle[].curves.n_curves)
            var n_area_lights = Int64(handle[].lights.n_area_lights)
            var n_distant_lights = Int64(handle[].lights.n_distant_lights)
            var n_infinite_lights = Int64(handle[].lights.n_infinite_lights)
            var n_point_lights = Int64(handle[].lights.n_point_lights)
            var n_blas = Int64(handle[].blas.n_blas)
            var n_instances = Int64(handle[].n_instances)
            var (spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) = handle[].spectral.unsafe_ptrs()
            var measured_brdfs = handle[].measured_brdfs_buf.unsafe_ptr().unsafe_bitcast[MeasuredBRDF_C]()
            var n_measured_brdfs = Int64(handle[].n_measured_brdfs)
            var gpu_textures = handle[].textures.textures_ptr()
            var n_gpu_textures = Int64(handle[].textures.n_textures)

            var grid_pix = ceildiv(n_pix, block_size)
            var grid_vps = ceildiv(n_vps, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            # Camera/visible-point samples are traced ONCE for the whole
            # render, not per SPPM pass — see _sppm_trace_visible_point's
            # docstring for why a per-pass re-trace breaks SPPM's
            # convergence guarantee.
            var cam_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
            handle[].ctx.enqueue_function[sppm_gen_vp_gpu](
                vps_ptr, inter_cam_ptr, Int64(n_pix), Int64(_VP_SAMPLES), psc[unsafe_offset=0].film_w, r2c_ptr, c2w_ptr,
                init_r2, cam_seed, Int64(psc[unsafe_offset=0].max_depth),
                film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                               psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                bvh2Nodes, primIds, meshes, materials,
                areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                gpu_textures, n_gpu_textures,
                grids_dev, Int64(handle[].n_grids), nvdb_grids_dev, Int64(handle[].n_nvdb_grids),
                grid_dim=grid_vps, block_dim=block_size)

            for pass_idx in range(n_passes):
                handle[].ctx.enqueue_function[sppm_reset_i32_gpu](
                    counter_ptr, grid_dim=1, block_dim=1)

                var pass_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
                var grid_emit = ceildiv(max(n_photons_per_pass, 1), block_size)
                handle[].ctx.enqueue_function[sppm_emit_photons_gpu](
                    photons_ptr, Int64(n_photons_per_pass), Int64(max_photons), inter_ph_ptr, counter_ptr,
                    default_emit_med, pass_seed, Int64(pass_idx), Int64(psc[unsafe_offset=0].max_depth),
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    c2w_ptr, photon_px_scale,
                    spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    gpu_textures, n_gpu_textures,
                    grids_dev, Int64(handle[].n_grids), nvdb_grids_dev, Int64(handle[].n_nvdb_grids),
                    grid_dim=grid_emit, block_dim=block_size)

                handle[].ctx.synchronize()
                var n_stored_raw: Int32
                with counter_buf.map_to_host() as host_buf:
                    var src = host_buf.unsafe_ptr().unsafe_bitcast[Int32]()
                    n_stored_raw = src[unsafe_offset=0]
                # A silent clamp is how dropped deposits stay invisible: the
                # estimator still divides by the FULL emitted count, so the
                # render just comes out patchy and dark with nothing in the
                # log. Saturation is a real failure mode here (it is what made
                # head.pbrt read ~33x before the buffer was sized for the
                # subsurface walk), so say so rather than absorbing it.
                if Int(n_stored_raw) > max_photons:
                    print("Warning: SPPM photon buffer saturated ("
                          + String(n_stored_raw) + " deposits into "
                          + String(max_photons) + " slots) — photons were dropped"
                          + " and this pass is biased dark. Raise --sppm-photons.")
                var n_stored = min(Int(n_stored_raw), max_photons)

                if n_stored > 0:
                    handle[].ctx.enqueue_function[sppm_grid_reset_gpu](
                        heads_ptr, Int64(_HSIZE), grid_dim=grid_hsize, block_dim=block_size)
                    var grid_ins = ceildiv(n_stored, block_size)
                    handle[].ctx.enqueue_function[sppm_grid_insert_gpu](
                        photons_ptr, Int64(n_stored), heads_ptr, inv_cell,
                        grid_dim=grid_ins, block_dim=block_size)
                    handle[].ctx.enqueue_function[sppm_gather_gpu](
                        vps_ptr, Int64(n_vps), photons_ptr, heads_ptr, inv_cell,
                        bvh2Nodes, primIds, meshes, materials, curves, n_curves, instances, n_instances,
                        spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                        measured_brdfs, n_measured_brdfs, Int64(pass_idx),
                        mediums, n_mediums, grids_dev, nvdb_grids_dev,
                        grid_dim=grid_vps, block_dim=block_size)

                var nee_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 0xBF58476D1CE4E5B9 + 3)
                handle[].ctx.enqueue_function[sppm_nee_gpu](
                    vps_ptr, Int64(n_vps), nee_seed, Int64(pass_idx),
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                    pointLights, n_point_lights,
                    spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                    measured_brdfs, n_measured_brdfs,
                    grid_dim=grid_vps, block_dim=block_size)

                if verbose or (pass_idx + 1) % 10 == 0:
                    print("SPPM (GPU): pass " + String(pass_idx + 1) + "/" + String(n_passes)
                          + " stored=" + String(n_stored), end="\r")

            print("")

            handle[].ctx.enqueue_function[sppm_finalize_gpu](
                vps_ptr, Int64(n_pix), Int64(_VP_SAMPLES), Int32(n_passes), iso_scale, max_comp, out_ptr, albedo_out_ptr,
                spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                grid_dim=grid_pix, block_dim=block_size)
            handle[].ctx.synchronize()
            # --- why-is-this-pixel-black diagnostic -------------------------
            # Four hypotheses about SPPM's remaining black pixels were refuted
            # in a row by guessing at mechanisms (delta bounces eating the VP
            # budget, the photon-pass equivalent, buffer saturation, glossy VP
            # placement). This says which term is actually zero instead.
            if verbose:
                var n_novp = 0; var n_nophot = 0; var n_dark = 0; var n_tot = 0; var n_envonly = 0; var n_bssrdf = 0; var n_bssrdf_lit = 0; var n_bssrdf_nan = 0
                with vps_buf.map_to_host() as vh:
                    var vp_host = vh.unsafe_ptr().unsafe_bitcast[SPPMPixel]()
                    for pi in range(n_pix):
                        var any_valid = False
                        var any_phot = False
                        var any_light = False
                        for s_i in range(_VP_SAMPLES):
                            var v = vp_host[unsafe_offset=pi * _VP_SAMPLES + s_i]
                            if v.valid != Int32(0):
                                any_valid = True
                                if v.N_acc > Float32(0): any_phot = True
                            if (v.ld.v0 + v.ld.v1 + v.ld.v2 + v.ld.v3) > Float32(1e-12): any_light = True
                            if (v.env.r + v.env.g + v.env.b) > Float32(1e-12): any_light = True
                        n_tot += 1
                        for s_i in range(_VP_SAMPLES):
                            var v2 = vp_host[unsafe_offset=pi * _VP_SAMPLES + s_i]
                            if v2.mat_kind == LobeKind.bssrdf and v2.valid != Int32(0):
                                n_bssrdf += 1
                                var ts = v2.tau.r + v2.tau.g + v2.tau.b
                                if ts > Float32(0): n_bssrdf_lit += 1
                                elif ts != ts: n_bssrdf_nan += 1
                        if not any_valid:
                            # Split the no-VP case: a camera ray that MISSES all
                            # geometry legitimately has no visible point and
                            # carries the environment in `env`. Only a pixel with
                            # neither a VP nor any env is genuinely dead.
                            if not any_light: n_novp += 1
                            else: n_envonly += 1
                        elif not any_phot and not any_light: n_dark += 1
                        elif not any_phot: n_nophot += 1
                var n_ph_surf = 0; var n_ph_vol = 0; var n_ph_bssrdf = 0
                with photons_buf.map_to_host() as ph_h:
                    var ph_host = ph_h.unsafe_ptr().unsafe_bitcast[SPPMPhoton]()
                    var n_scan = min(max_photons, 200000)
                    for k in range(n_scan):
                        var kind = Int(ph_host[unsafe_offset=k].is_volume)
                        if kind == 0: n_ph_surf += 1
                        elif kind == 1: n_ph_vol += 1
                        elif kind == 2: n_ph_bssrdf += 1
                print("SPPM diag photons (last pass, first " + String(min(max_photons,200000))
                      + " slots): surface=" + String(n_ph_surf) + " volume=" + String(n_ph_vol)
                      + " bssrdf=" + String(n_ph_bssrdf))
                print("SPPM diag: " + String(n_tot) + " pixels | DEAD (no VP, no env): " + String(n_novp)
                      + " | no VP but env only: " + String(n_envonly)
                      + " | VP but zero photons AND no light: " + String(n_dark)
                      + " | VP with light but zero photons: " + String(n_nophot)
                      + " || BSSRDF VPs: " + String(n_bssrdf) + " of which tau>0: " + String(n_bssrdf_lit) + " NaN: " + String(n_bssrdf_nan))
            # Keep these device buffers alive (Mojo's ASAP destruction would
            # otherwise free them right after their own last syntactic
            # reference, which is BEFORE this point -- their derived _ptr
            # pointers, widened to MutUntrackedOrigin for the enqueue_function
            # calls above, carry no lifetime tracking back to the owning
            # buffer, so the buffer's own reference is what has to survive
            # until every kernel that could touch its memory has completed,
            # i.e. past this synchronize()) -- a real GPU-memory
            # use-after-free risk if removed, not a style nicety.
            _ = vps_buf^; _ = photons_buf^; _ = heads_buf^
            _ = inter_cam_buf^; _ = inter_ph_buf^; _ = r2c_buf^; _ = c2w_buf^

            var out_pixels = unsafe_alloc[Float32](n_pix * 3)
            with out_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = out_pixels.unsafe_bitcast[UInt8]()
                for i in range(n_pix * 3 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            # Denoise (never wired up before -- no_denoise was a dead
            # parameter): read back the albedo AOV finalized above, run a
            # fresh normals/depth pass via the host-side sd (same
            # render_aux_buffers the CPU path/plain tracer use), then the
            # same CPU denoise() the CPU SPPM path uses.
            var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
            with albedo_out_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = albedo_pixels.unsafe_bitcast[UInt8]()
                for i in range(n_pix * 3 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            var normals = unsafe_alloc[Float32](n_pix * 3)
            var depth = unsafe_alloc[Float32](n_pix)
            var sd_local = sd
            render_aux_buffers(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world, Int32(0), Int32(0),
                                psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h, Pointer(to=sd_local), normals, depth)

            var denoised = unsafe_alloc[Float32](n_pix * 3)
            if no_denoise:
                for i in range(n_pix * 3): denoised[unsafe_offset=i] = out_pixels[unsafe_offset=i]
            else:
                denoise(out_pixels, albedo_pixels, normals, depth, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
                        denoised, Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

            _ = write_image_cropwindow(denoised, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
                psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
                psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))
            out_pixels.unsafe_free(); albedo_pixels.unsafe_free(); normals.unsafe_free(); depth.unsafe_free(); denoised.unsafe_free()
        except e:
            print("SPPM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("SPPM GPU: no accelerator")
        ret = Int32(-1)
    return ret
