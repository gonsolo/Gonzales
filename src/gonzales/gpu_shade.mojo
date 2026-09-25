from .bvh import BVH2Node, SceneDescriptor2_C
from .curves import Curve_C
from .geometry import _is_real_ptr
from .guide import null_guide
from .lights import AreaLight, DistantLight, InfiniteLight, LightSampler, PointLight
from .materials import MatKind, Material_C, MeasuredBRDF_C
from .media import MediumInterface
from .primitives import Instance, Intersection, PrimId, Sphere, TriangleMesh
from .render_state import GpuTexture_C, NormalSlopeMap_C, PathState_C, ShadowTask_C
from .restir_di import DIReservoir, ReservoirIO, reservoir_io_null
from .restir_gi import gi_reservoir_io_null
from .rng import PCG32
from .shading import GIPendingX1, LightContext, ShadeContext, shade_coated_conductor, shade_coated_diffuse, shade_conductor, shade_core, shade_dielectric, shade_diffuse, shade_diffuse_transmission, shade_hair, shade_interface, shade_measured, shade_nee_core, shade_thin_dielectric
from .spectrum import SpectralHandle
from max.gpu import block_dim, block_idx, thread_idx
from .gpu_scene import GpuSceneHandle

@always_inline
def _shade_context(
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    px_scale: Float32,
    path_idx: Int = 0,
    use_restir: Bool = False,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin] = Pointer[ShadowTask_C, MutUntrackedOrigin].unsafe_dangling(),
) -> ShadeContext:
    return ShadeContext(
        path_idx=path_idx, bvh2Nodes=sd.bvh2Nodes, primIds=sd.primIds, meshes=sd.meshes, curves=sd.curves,
        materials=sd.materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=sd.gpuTextures, n_textures=Int(sd.gpuTextureCount),
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=use_restir,
        blasNodesArr=sd.blasNodesArr, blasPrimIdsArr=sd.blasPrimIdsArr, instances=sd.instances,
        spectral=sd.spectral, measured_brdfs=sd.measuredBrdfs,
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=sd.areaLights, area_light_count=Int(sd.areaLightCount),
            distant_lights=sd.distantLights, distant_count=Int(sd.distantLightCount),
            point_lights=sd.pointLights, point_count=Int(sd.pointLightCount),
            infinite_lights=sd.infiniteLights, infinite_count=Int(sd.infiniteLightCount),
            spheres=sd.spheres, sphere_count=Int(sd.sphereCount), light_sampler=sd.lightSampler))


def shade_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    spectral: SpectralHandle,
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shade_core(paths, intersections, meshes, materials, spectral, tid)



def shade_nee_preamble_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    var ctx_no_shadow = _shade_context(sd, sobol_matrices, px_scale)
    shade_nee_core[True, False](path_ptr, inter, ctx_no_shadow)


# ── Per-material GPU kernels (G1) ─────────────────────────────────────────────
# shade_nee_preamble_gpu handles miss + emission, then sets pending_mat.
# Each kernel below checks pending_mat, clears it, and calls the shade function.

def shade_diffuse_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    # ReSTIR DI (Phase 2, --restir). Only gpu_render_sample ever passes
    # use_restir=True here -- gpu_render_wavefront has no ReSTIR concept at
    # all (see its own docstring: batch --restir renders via
    # gpu_render_sample instead, precisely to avoid the
    # WAVEFRONT_BATCH-concurrent-samples-per-pixel problem). All defaulted-
    # inert so gpu_render_wavefront's dispatch is unaffected.
    # Int32 rather than Bool: GPU kernel arguments must be DevicePassable and
    # Bool is not, which the compiler only reports at the enqueue site.
    use_restir: Int32 = Int32(0),
    restir_read: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    restir_write: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_normal: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_material_id: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    frame_w: Int32 = Int32(0),
    frame_h: Int32 = Int32(0),
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var restir_on = use_restir != Int32(0)
    var ctx = _shade_context(sd, sobol_matrices, px_scale, use_restir=restir_on)
    # tid IS the pixel index here: this kernel only ever sees use_restir=True
    # from gpu_render_sample, which runs exactly one path per pixel (see the
    # param block above -- gpu_render_wavefront never sets it). restir_on
    # without real buffers (non-restir renders) still needs pixel_idx=-1,
    # di_temporal_step's own "no reuse" sentinel.
    var restir_has_state = restir_on and _is_real_ptr(restir_read)
    var restir_io = reservoir_io_null()
    if restir_has_state:
        restir_io = ReservoirIO(
            read=restir_read, write=restir_write,
            gbuf_normal=gbuf_normal, gbuf_depth=gbuf_depth,
            gbuf_material_id=gbuf_material_id, gbuf_world_pos=gbuf_world_pos,
            frame_w=frame_w, frame_h=frame_h)
    shade_diffuse[True, False](path_ptr, inter, ctx, mat, null_guide(), restir_io, tid if restir_has_state else -1)


def shade_coated_diffuse_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.coated_diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ctx = _shade_context(sd, sobol_matrices, px_scale, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_coated_diffuse[True, False](path_ptr, inter, ctx, mat)


def shade_diffuse_transmit_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.diffuse_transmit:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var ctx = _shade_context(sd, sobol_matrices, px_scale, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_diffuse_transmission[True, False](path_ptr, inter, ctx)


# GPU-only: mix is a pure material SELECTOR, not a shader -- it has no BSDF,
# no NEE, no light/texture/spectral dependence of its own. The old
# implementation called shade_mix[True,False] with a full ShadeContext
# (33 parameters, mirroring every real per-material kernel's signature) which
# then called the @always_inline _shade_dispatch -- inlining the FULL shading
# code of every OTHER material (diffuse, conductor, dielectric,
# coated_diffuse, diffuse_transmission, coated_conductor, thin_dielectric,
# interface, measured) into this one function. That made shade_mix_gpu's
# compiled body easily the largest function in the codebase, and this
# machine's CUDA 13.3 driver / Modular 26.4.0 toolchain cannot produce valid
# PTX for it (confirmed via kernel-by-kernel bisection: shade_mix_gpu alone
# reproduces CUDA_ERROR_INVALID_PTX; every other kernel, including
# shade_measured_gpu, compiles and runs fine without it).
#
# Fix: don't shade anything here at all. Pick the sub-material (same RNG
# draw + mix-of-mix guard as shade_mix, shading.mojo) and redirect --
# overwrite this hit's materialIndex to the CHOSEN sub-material (still an
# index into the same `sd.materials` array) and re-tag pending_mat with the
# sub-material's real type, exactly mirroring what shade_nee_core's own
# GPU branch does for every material ("mark material for its dedicated
# per-material kernel"). CUDA kernels launched on one stream execute in
# launch order, so as long as this kernel is enqueued BEFORE every other
# per-material kernel in the same bounce's dispatch sequence (see both
# gpu_render_sample/gpu_render_wavefront call sites), the sub-material's
# real kernel picks up the redirected pending_mat/materialIndex later in
# this SAME pass and shades it with its own full NEE/BSDF logic --
# identical end result to the CPU path, just via a two-step handoff instead
# of one big inlined function. This can't be deferred to the NEXT bounce:
# the ray is never advanced here, so a fresh intersection at the start of
# the next bounce would just re-hit the same mix material and re-roll the
# choice forever.
def shade_mix_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.mix:
        return
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var packed = mat.tex_idx
    var idx1 = Int(packed & Int32(0xFFFF))
    var idx2 = Int((packed >> 16) & Int32(0xFFFF))
    var amount = mat.roughU  # blend factor: 0 = all mat1, 1 = all mat2
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var chosen_idx = idx2 if pcg.next_float() < amount else idx1
    path_ptr[].pcgState = pcg.state
    var sub_type = sd.materials[unsafe_offset=chosen_idx].type
    if sub_type == MatKind.mix:
        sub_type = MatKind.diffuse  # guard against mix-of-mix cycle, matches shade_mix (shading.mojo)
    intersections[unsafe_offset=tid].primId.materialIndex = Int64(chosen_idx)
    path_ptr[].pending_mat = sub_type


def shade_conductor_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ctx = _shade_context(sd, sobol_matrices, px_scale, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_measured_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.measured:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ctx = _shade_context(sd, sobol_matrices, px_scale, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_measured[True, False](path_ptr, inter, ctx, mat)


def shade_dielectric_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    count_dp: Int64,
    px_scale: Float32,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    # sd.gpuTextures/px_scale are here only so a dielectric carrying "texture
    # displacement"/"normalmap" gets it applied (barcelona-pavilion's water).
    # tex_filenames is CPU-only (GPU samples the uploaded texture table), so
    # the dangling default is correct on this path.
    shade_dielectric[True](path_ptr, inter, sd.meshes, mat, sd.spheres,
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        sd.gpuTextures, Int(sd.gpuTextureCount), px_scale)


def shade_thin_dielectric_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.thin_dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    shade_thin_dielectric(path_ptr, inter, sd.meshes, mat, sd.spheres)


def shade_coated_conductor_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.coated_conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ctx = _shade_context(sd, sobol_matrices, px_scale, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_coated_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_interface_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    count_dp: Int64,
):
    """Passthrough (interface) material: advance ray through the surface.
    Medium update is handled by update_medium_gpu which runs after all shaders."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.interface:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    shade_interface(path_ptr, inter)


def shade_hair_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.hair:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ctx = _shade_context(sd, sobol_matrices, px_scale, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_hair[True, False](path_ptr, inter, ctx, mat)


def shade_enqueue_shadow_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight, MutUntrackedOrigin],
    areaLightCount: Int,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures: Int,
    infiniteLights: Pointer[InfiniteLight, MutUntrackedOrigin],
    n_infinite_lights: Int,
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    n_spheres: Int,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    count: Int,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shadow_tasks[unsafe_offset=tid].active = Int32(0)
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    var ls_shadow = LightSampler(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), Int32(0))
    var ctx_shadow = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin](),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=Float32(0.0), sobol_matrices=Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling(), guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=Pointer[DistantLight, MutUntrackedOrigin](), distant_count=0,
            point_lights=Pointer[PointLight, MutUntrackedOrigin](), point_count=0,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls_shadow))
    shade_nee_core[True, True](path_ptr, inter, ctx_shadow)


# Phase 0.4 (docs/A2_restir_migration_plan.md): a task deferred by one
# material's per-pixel kernel this bounce (enqueue_shadow=True) must not be
# resolved twice, and a pixel shaded by a material that did NOT defer (still
# resolves its own shadow ray inline) must not have a stale prior-bounce
# task resolved in its place -- both need shadow_tasks[tid].active reset to
# 0 before any of this bounce's per-material kernels run, since only the one
# kernel matching pending_mat[tid] actually touches slot tid.
