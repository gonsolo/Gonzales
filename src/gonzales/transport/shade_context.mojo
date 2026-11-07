"""What a shading point can see of the scene."""
from gonzales.core.geometry import RGB, Vec3f
from gonzales.scene.appearance.materials import Material, MeasuredBRDF
from gonzales.transport.render_state import PathState, ShadowTask, SuppressTask
from gonzales.scene.appearance.materials import GpuTexture, NormalSlopeMap
from gonzales.scene.shapes.primitives import PrimId, TriangleMesh, Sphere, Instance
from gonzales.scene.appearance.lights import AreaLight, DistantLight, PointLight, InfiniteLight, LightSampler
from gonzales.scene.appearance.media import Medium, MediumInterface, Grid, NvdbGrid
from gonzales.scene.shapes.curves import Curve
from gonzales.scene.appearance.footprint import CameraFootprint
from gonzales.scene.accel.bvh import BVH2Node, BVH4, SceneView
from gonzales.transport.guide import GuideGrid, null_guide
from gonzales.core.spectrum import SpectralHandle, SpectralSample
from gonzales.transport.restir_gi import GIReservoirIO, gi_reservoir_io_null


@fieldwise_init
struct GIPendingX1(TrivialRegisterPassable):
    """Phase 4's per-path-slot scratch carrying x1's own shading data forward
    from bounce 0 to bounce 1, so bounce 1 (if it also lands on a diffuse
    vertex) can weight its fresh GIReservoir candidate via gi_target_pdf --
    which needs x1's hit_point/normal/alb, no longer in scope by the time
    bounce 1's own _shade_diffuse_nee call runs. `active` distinguishes "no
    GI candidate pending" (0, the common case: bounce 0 wasn't diffuse, or
    ctx.use_restir is off) from "pending, awaiting bounce 1" (1) -- cleared
    back to 0 once bounce 1 consumes it (or left at 1 forever, harmlessly,
    if the path terminates/misses/hits non-diffuse before bounce 1, since
    this buffer's lifetime is exactly one render_tile call and is never
    reused across samples -- see rendering.mojo's allocation site).

    `throughput` MUST be captured here (path_ptr[].throughput at bounce 0,
    BEFORE shade_diffuse's own continuation-sampling epilogue multiplies it
    by x1's OWN real BSDF sample) -- reading path_ptr[].throughput fresh at
    bounce 1 instead would pick up x1's ORIGINAL sampled direction's
    f/cos/pdf factors, which have nothing to do with the RESAMPLED
    reconnection direction gi_resolve actually connects along. This was a
    real, shipped bug (~10x energy inflation, caught only once interactive
    mode's real per-bounce throughput evolution was exercised -- the
    earlier end-to-end unit test used a synthetic throughput=1 and never
    called the real shade_diffuse epilogue between bounces, so it couldn't
    have caught this)."""
    var active:     Int8
    var hit_point:  Vec3f
    var normal:     Vec3f
    var alb:        RGB
    var throughput: SpectralSample


@fieldwise_init
struct LightContext(Copyable, Movable):
    """Light source arrays + counts + sampler, used by NEE. Split out of
    ShadeContext (step 7) so the grouping of "which lights exist in the scene"
    is self-contained instead of interleaved with scene/texture/sampling
    fields — construction call sites build this with keyword args precisely
    because several fields share the same Int/pointer type and a positional
    transposition wouldn't be caught by the type checker."""
    var area_lights:      Pointer[AreaLight, MutUntrackedOrigin]
    var area_light_count: Int
    var distant_lights:   Pointer[DistantLight, MutUntrackedOrigin]
    var distant_count:    Int
    var point_lights:     Pointer[PointLight, MutUntrackedOrigin]
    var point_count:      Int
    var infinite_lights:  Pointer[InfiniteLight, MutUntrackedOrigin]
    var infinite_count:   Int
    var spheres:          Pointer[Sphere, MutUntrackedOrigin]
    var sphere_count:     Int
    var light_sampler:    LightSampler


@fieldwise_init
struct ShadeContext:
    """Bundles scene/texture/sampling data pointers passed to shade_nee_core
    and material shaders. Light-source data lives in the nested `lights`
    LightContext (step 7)."""
    var path_idx:         Int
    var bvh2Nodes:        Pointer[BVH2Node, MutUntrackedOrigin]
    var primIds:          Pointer[PrimId, MutUntrackedOrigin]
    var meshes:           Pointer[TriangleMesh, MutUntrackedOrigin]
    var curves:           Pointer[Curve, MutUntrackedOrigin]
    var materials:        Pointer[Material, MutUntrackedOrigin]
    var tex_filenames:    Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var textures:         Pointer[GpuTexture, MutUntrackedOrigin]
    var n_textures:       Int
    # Slope-space copies of the normal maps, indexed exactly like
    # `tex_filenames`/`textures` (so a Material's `normal_tex_idx`
    # addresses all three). Read ONLY by the SMS/MNEE manifold walk, which
    # needs the normal's derivatives and not just the normal -- see
    # geometry.mojo's NormalSlopeMap. Dangling on the GPU call sites (the
    # device-side scene upload has no slope maps yet, so a normal-mapped
    # caustic caster falls back to its smooth surface there, exactly as it
    # did before these existed).
    var nmaps:            Pointer[NormalSlopeMap, MutUntrackedOrigin]
    var shadow_tasks:     Pointer[ShadowTask, MutUntrackedOrigin]
    var cam_fp:           CameraFootprint   # pbrt texture/bump footprint (footprint.mojo)
    var sobol_matrices:   Pointer[UInt32, MutUntrackedOrigin]
    var guide:            GuideGrid
    var guide_write:      GuideGrid         # GPU: the tree this iteration records into (CPU passes it as a shade argument)
    # Phase 2 (docs/A2_restir_migration_plan.md): CPU-only today, like
    # `guide` above -- every GPU ShadeContext construction site passes
    # False. When True, the diffuse material's area-light NEE (bounce 0
    # only) is replaced by RIS candidate generation + a single deferred
    # shadow-ray resolve (restir_di.mojo) instead of one direct NEE sample.
    var use_restir:       Bool
    var lights:           LightContext
    # Object instancing (see geometry.mojo's Instance docs). GPU call sites
    # pass dangling/zero-count values — GPU's own device-side scene upload
    # never populates instance data, so no PrimId.type==6 leaf can appear
    # there and these are never dereferenced on that path.
    var blasNodesArr:   Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin]
    var blasPrimIdsArr: Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin]
    var instances:      Pointer[Instance, MutUntrackedOrigin]
    # Staged spectral rendering rollout (Stage 2c, see project_spectral_rendering
    # memory) — host pointers on CPU, device pointers on GPU (see gpu.mojo's
    # ShadeContext construction sites).
    var spectral:       SpectralHandle
    # "measured" materials: one MeasuredBRDF per distinct .bsdf file,
    # indexed by Material.measured_idx (see measured_bsdf.mojo's loader).
    # Host pointer on CPU; dangling on GPU call sites until Stage 3 uploads
    # these as device buffers.
    var measured_brdfs: Pointer[MeasuredBRDF, MutUntrackedOrigin]
    # Phase 4 (docs/A2_restir_migration_plan.md), CPU-only, dead/unwired
    # (dangling/null on every call site except shade_core_cpu_nee's, which
    # is itself not called with real buffers by any render path yet -- see
    # render_tile/pipeline.mojo). `gi_pending` is per-PATH-SLOT scratch
    # (indexed by ctx.path_idx, lifetime = one render_tile call): whether
    # bounce 0 landed on a diffuse x1 with restir active, plus x1's own
    # hit_point/normal/alb needed to weight the fresh candidate once bounce
    # 1 resolves it (see GIPendingX1's docstring). `gi_io` is the frame-
    # wide, caller-owned read/write reservoir buffers + shared G-buffer
    # (same shape as restir_di.mojo's ReservoirIO, reused as-is for spatial
    # rejection since G-buffer data is a property of the pixel's primary
    # hit, not of which reservoir type is being reused) that
    # gi_temporal_spatial_combine/gi_resolve consume once bounce 1 lands on
    # a diffuse x2 too (this scope's restriction, see
    # _gi_generate_recon_candidate's docstring for why).
    var gi_pending: Pointer[GIPendingX1, MutUntrackedOrigin]
    var gi_io:      GIReservoirIO
    # GPU kernels that share ONE device-resident context between all threads (no per-thread copy on the stack) cannot
    # carry the path index in it; the base of the path array lets _shadow_contribute recover it from path_ptr.
    # Dangling = the context is per-thread and path_idx is authoritative.
    var paths_base: Pointer[PathState, MutUntrackedOrigin]
    # False when the scene has no smooth glass: MNEE's per-sample probe toward the light can then never find anything.
    var has_glass: Bool
    var glossy_sms: Bool   # smooth glass glossy SMS may reach (stays on when MNEE's probes switch has_glass off)
    # GPU only: [MNEE probes, probes that found glass] counted on 1 call in 16, so the host can switch MNEE off when glass is rare; dangling = not counting.
    var mnee_stats: Pointer[Int32, MutUntrackedOrigin]
    # CPU only: 4-wide BVH for the any-hit / closest-hit calls below; BVH4.none() = BVH2 (always on the GPU).
    var bvh4: BVH4
    # GPU only: where the emitter-hit code parks a glass-chain hit for resolve_suppress_gpu; dangling = decide inline
    var suppress_tasks: Pointer[SuppressTask, MutUntrackedOrigin]
    # Participating media for surface-NEE shadow rays; n_mediums == 0 keeps the plain any-hit test.
    var n_mediums:     Int
    var mediums:       Pointer[Medium, MutUntrackedOrigin]
    var medium_ifaces: Pointer[MediumInterface, MutUntrackedOrigin]
    var grids:         Pointer[Grid, MutUntrackedOrigin]
    var nvdb_grids:    Pointer[NvdbGrid, MutUntrackedOrigin]
    # Cycles' Filter Glossy (scene.cycles.blur_glossy); 0 = off
    var filter_glossy: Float32
    var clamp: SIMD[DType.float32, 2]   # SceneView.clamp
    var cycles: SIMD[DType.int32, 4]    # SceneView.cycles


@always_inline
def _shade_context(
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    path_idx: Int = 0,
    use_restir: Bool = False,
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin] = Pointer[ShadowTask, MutUntrackedOrigin].unsafe_dangling(),
    suppress_tasks: Pointer[SuppressTask, MutUntrackedOrigin] = Pointer[SuppressTask, MutUntrackedOrigin].unsafe_dangling(),
) -> ShadeContext:
    """A ShadeContext over a SceneView: the GPU kernels' and VCM's way to reach the shading code."""
    return ShadeContext(
        paths_base=Pointer[PathState, MutUntrackedOrigin].unsafe_dangling(), has_glass=(sd.hasGlass & Int32(1)) != Int32(0), glossy_sms=(sd.hasGlass & Int32(2)) != Int32(0), mnee_stats=Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), bvh4=BVH4.none(), suppress_tasks=suppress_tasks,
        n_mediums=Int(sd.mediumCount), mediums=sd.mediums, medium_ifaces=sd.mediumInterfaces, grids=sd.grids, nvdb_grids=sd.nvdbGrids,
        path_idx=path_idx, bvh2Nodes=sd.bvh2Nodes, primIds=sd.primIds, meshes=sd.meshes, curves=sd.curves,
        materials=sd.materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=sd.gpuTextures, n_textures=Int(sd.gpuTextureCount),
        nmaps=Pointer[NormalSlopeMap, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        cam_fp=sd.camFp, sobol_matrices=sobol_matrices, guide=null_guide(), guide_write=null_guide(), use_restir=use_restir,
        blasNodesArr=sd.blasNodesArr, blasPrimIdsArr=sd.blasPrimIdsArr, instances=sd.instances,
        spectral=sd.spectral, measured_brdfs=sd.measuredBrdfs,
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=sd.areaLights, area_light_count=Int(sd.areaLightCount),
            distant_lights=sd.distantLights, distant_count=Int(sd.distantLightCount),
            point_lights=sd.pointLights, point_count=Int(sd.pointLightCount),
            infinite_lights=sd.infiniteLights, infinite_count=Int(sd.infiniteLightCount),
            spheres=sd.spheres, sphere_count=Int(sd.sphereCount), light_sampler=sd.lightSampler), filter_glossy=sd.filterGlossy, clamp=sd.clamp, cycles=sd.cycles)
