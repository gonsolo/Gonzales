from gonzales.scene.appearance.light_tree import light_tree_empty
from gonzales.core.spectrum import null_spectral_handle
from gonzales.scene.accel.bvh import BVH4, BVH2Node
from gonzales.scene.accel.xert import xe_scene_none
from gonzales.scene.shapes.motion import MotionTable
from gonzales.scene.appearance.footprint import CameraFootprint
from gonzales.scene.appearance.lights import LightSampler, AreaLight, DistantLight, PointLight, InfiniteLight
from gonzales.scene.appearance.materials import Material, MeasuredBRDF, GpuTexture, NormalSlopeMap
from gonzales.scene.appearance.media import Medium, MediumInterface, Grid, NvdbGrid
from gonzales.scene.shapes.curves import Curve
from gonzales.scene.shapes.primitives import PrimId, TriangleMesh, Instance, Sphere
from gonzales.transport.guide import null_guide
from gonzales.transport.render_state import ShadowTask, SuppressTask, PathState
from gonzales.transport.restir_gi import gi_reservoir_io_null
from gonzales.transport.shade_context import ShadeContext, LightContext, GIPendingX1

# ── Shared ShadeContext fixture ─────────────────────────────────────────────
# Every field a test does not pass is a dangling sentinel or off, so a new
# ShadeContext field is added here once instead of in every test.


def null_light_context() -> LightContext:
    return LightContext(
        Pointer[AreaLight, MutUntrackedOrigin].unsafe_dangling(), 0,
        Pointer[DistantLight, MutUntrackedOrigin].unsafe_dangling(), 0,
        Pointer[PointLight, MutUntrackedOrigin].unsafe_dangling(), 0,
        Pointer[InfiniteLight, MutUntrackedOrigin].unsafe_dangling(), 0,
        Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(), 0,
        LightSampler(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), light_tree_empty()),
    )


def make_shade_context(
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    var lights: LightContext,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin] = Pointer[BVH2Node, MutUntrackedOrigin].unsafe_dangling(),
    primIds: Pointer[PrimId, MutUntrackedOrigin] = Pointer[PrimId, MutUntrackedOrigin].unsafe_dangling(),
    materials: Pointer[Material, MutUntrackedOrigin] = Pointer[Material, MutUntrackedOrigin].unsafe_dangling(),
    use_restir: Bool = False,
    gi_pending: Pointer[GIPendingX1, MutUntrackedOrigin] = Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(),
) -> ShadeContext:
    return ShadeContext(
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes,
        curves=Pointer[Curve, MutUntrackedOrigin].unsafe_dangling(),
        materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=Pointer[GpuTexture, MutUntrackedOrigin].unsafe_dangling(), n_textures=0,
        nmaps=Pointer[NormalSlopeMap, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=Pointer[ShadowTask, MutUntrackedOrigin].unsafe_dangling(),
        cam_fp=CameraFootprint.none(),
        sobol_matrices=Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling(),
        guide=null_guide(), guide_write=null_guide(),
        use_restir=use_restir,
        lights=lights^,
        blasNodesArr=Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        blasPrimIdsArr=Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        instances=Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
        spectral=null_spectral_handle(),
        measured_brdfs=Pointer[MeasuredBRDF, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=gi_pending, gi_io=gi_reservoir_io_null(),
        paths_base=Pointer[PathState, MutUntrackedOrigin].unsafe_dangling(), has_glass=True, glossy_sms=True,
        bvh4=BVH4.none(),
        suppress_tasks=Pointer[SuppressTask, MutUntrackedOrigin].unsafe_dangling(),
        gsms_queue=Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
        n_mediums=0, mediums=Pointer[Medium, MutUntrackedOrigin].unsafe_dangling(),
        medium_ifaces=Pointer[MediumInterface, MutUntrackedOrigin].unsafe_dangling(),
        grids=Pointer[Grid, MutUntrackedOrigin].unsafe_dangling(), nvdb_grids=Pointer[NvdbGrid, MutUntrackedOrigin].unsafe_dangling(),
        filter_glossy=Float32(0), clamp=SIMD[DType.float32, 2](0), cycles=SIMD[DType.int32, 4](0),
        xe=xe_scene_none(),
        motion=MotionTable.none(),
    )
