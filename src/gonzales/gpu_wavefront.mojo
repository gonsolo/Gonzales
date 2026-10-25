from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.os import getenv
from std.time import perf_counter_ns
from std.utils import StaticTuple
from .gpu_tuning import MINCTA_TRAVERSE
from std.sys import get_defined_int
from .bvh import _node_box_hit, _node_hi, _node_lo, _node_oc, _bvh_has_nodes, _shadow_is_null_material, _traverse_instances, BVH2Node, any_hit_bvh2_core, ray_sphere_hit, test_spheres, traverse_bvh2_core, traverse_bvh2_core_defer_curves, SceneView
from .curve_bvh import curve_bvh_occluded, curve_bvh_closest, curve_bvh_prim
from .curves import CURVE_DEFER_K, Curve, _curve_perp_axis, curve_piece_endpoints, intersect_curve, curve_hit_normal
from .geometry import INV_FOUR_PI, Point3f, RGB, Vec3f, _is_real_ptr, cross, dot, store_vec3, vec3f
from .materials import Material
from .primitives import Instance, Intersection, PrimId, Ray, Sphere, TriangleMesh, alpha_killed, intersect_triangle, sphere_outward_normal, mesh_vi
from .primitives import mesh_pos, mesh_point, mesh_material
from .render_state import PathState, MIN_RAY_PDF_NONE, SssWalk, ShadowTask, SHADOW_SLOTS
from .restir_di import DIReservoir, di_reservoir_init
from .restir_vol import VolReservoir, vol_reservoir_init
from .sampling import gen_primary_ray_state, ThinLens
from .spectrum import SpectralSample, spectral_sample_to_rgb
from .transform import transform_normal, Mat4
from .vulkaninterop import VulkanInteropRtSceneHandle, vulkaninterop_rt_trace
from .rtcore import rtcore_active, rtcore_alpha_enabled, rtcore_trace, rtcore_trace_interop
from .optixcurves import oxc_active, oxc_scratch, oxc_trace_closest, oxc_trace_shadow
from max.gpu import WARP_SIZE, barrier, block_dim, block_idx, lane_id, thread_idx
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from max.gpu.primitives.warp import shuffle_idx, vote
from std.bit import count_trailing_zeros, pop_count
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host._nvidia_cuda import CUDA
from std.atomic import Atomic
from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from .gpu_scene import GpuSceneHandle
from .gpu_media import _sss_walk_step

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def reset_shadow_tasks_gpu(
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    count_dp: Int64,
    slots_dp: Int64 = Int64(1),
    usable_dp: Int64 = Int64(1),
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    # A slot beyond the usable ones is marked taken (-1), so _shadow_contribute traces that candidate inline.
    for s in range(Int(slots_dp)):
        shadow_tasks[unsafe_offset=tid * Int(slots_dp) + s].active = Int32(0) if s < Int(usable_dp) else Int32(-1)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def reset_restir_reservoirs_gpu(
    reservoirs: Pointer[DIReservoir, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Clear ReSTIR DI reservoirs to "no candidate yet". Needed at scene
    upload and on every camera move: identity reprojection assumes the
    previous frame's reservoir describes THIS pixel's shading point, so a
    surviving reservoir after the camera moves would reuse a light chosen
    for a different view. The GPU film is cleared on the same events for
    exactly the same reason (gpu_clear_film)."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    reservoirs[unsafe_offset=tid] = di_reservoir_init()

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def reset_restir_vol_reservoirs_gpu(
    reservoirs: Pointer[VolReservoir, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Clear volume ReSTIR reservoirs to "no candidate yet" -- the same
    identity-reprojection invalidation reason as reset_restir_reservoirs_gpu
    above, applied to Phase 7.3's per-pixel volume-scatter reservoirs."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    reservoirs[unsafe_offset=tid] = vol_reservoir_init()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def reset_vol_used_gpu(
    used: Pointer[Int8, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Zero the per-pixel "already combined this frame" guard -- called once
    at the start of every gpu_render_sample dispatch, NOT on camera move
    (unlike reset_restir_vol_reservoirs_gpu above): this buffer has no
    cross-frame meaning at all, it only disambiguates within ONE frame's own
    bounce-round loop. See _sample_medium_core's vol_used comment."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    used[unsafe_offset=tid] = Int8(0)


def traverse_shadow_rays_gpu(
    sd: SceneView,
    paths: Pointer[PathState, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    count_dp: Int64,
):
    var n_spheres = Int(sd.sphereCount)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var task = shadow_tasks[unsafe_offset=tid]
    if task.active == 0:
        return
    var shadow_ray = Ray(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
    if not any_hit_bvh2_core[True](sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray, task.tmax, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, n_spheres, materials=sd.materials):
        paths[unsafe_offset=tid].estimate += task.contrib



@always_inline
def _clamp_sample_rgb(r: Float32, g: Float32, b: Float32, lim: Float32
                      ) -> Tuple[Float32, Float32, Float32]:
    """pbrt's `maxcomponentvalue`, applied to ONE SAMPLE as pbrt applies it.

    RGBFilm::AddSample clamps each sample's sensor RGB before it enters the
    filter accumulator. normalize_film used to be our only clamp, and it runs
    on the FINISHED pixel -- a 64-sample mean almost never crosses the
    threshold, so the clamp removed essentially nothing while pbrt was
    removing real energy from every spiky sample. Measured on
    barcelona-pavilion, where the scene asks for maxcomponentvalue 50 at iso
    500: the lit deck read 1.410x the reference, and 1.197x with the clamp
    taken out of BOTH renderers -- i.e. this asymmetry owned most of that gap.

    `lim` is already divided by the film's iso scale, because normalize_film
    multiplies by iso/100 after the fact and pbrt clamps post-sensor. The
    test is on max(X, Y, Z), pbrt's sensor space -- see RGB.sensor_clamped.
    lim <= 0 disables it."""
    var c = RGB(r, g, b).sensor_clamped(lim)
    return (c.r, c.g, c.b)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def accumulate_film_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
    film_sq: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
    sample_clamp: Float32 = Float32(0.0),
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    # ── Output boundary: spectral transport -> RGB film ──────────────────
    var _e = spectral_sample_to_rgb(spectral_coeffs, Int(spectral_res_dp),
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        paths[unsafe_offset=tid].estimate, paths[unsafe_offset=tid].wavelengths)
    var (_cr, _cg, _cb) = _clamp_sample_rgb(_e[0], _e[1], _e[2], sample_clamp)
    film[unsafe_offset=tid*3+0] += _cr
    film[unsafe_offset=tid*3+1] += _cg
    film[unsafe_offset=tid*3+2] += _cb
    var _ql = RGB(_cr, _cg, _cb).luma()
    film_sq[unsafe_offset=tid] += _ql * _ql
    albedo_film[unsafe_offset=tid*3+0] += paths[unsafe_offset=tid].albedo.r
    albedo_film[unsafe_offset=tid*3+1] += paths[unsafe_offset=tid].albedo.g
    albedo_film[unsafe_offset=tid*3+2] += paths[unsafe_offset=tid].albedo.b


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def clear_film_gpu(film: Pointer[Float32, MutUntrackedOrigin], n_pixels_dp: Int64):
    var n_pixels = Int(n_pixels_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    film[unsafe_offset=tid*3+0] = Float32(0)
    film[unsafe_offset=tid*3+1] = Float32(0)
    film[unsafe_offset=tid*3+2] = Float32(0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def clear_scalar_film_gpu(film: Pointer[Float32, MutUntrackedOrigin], n_pixels_dp: Int64):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(n_pixels_dp):
        return
    film[unsafe_offset=tid] = Float32(0)


# Wavefront accumulation: thread px sums actual_batch samples from path_buf layout
# path_buf[si * n_pixels + px] and adds to film[px].  No atomics needed (one thread per pixel).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def accumulate_film_wavefront_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
    film_sq: Pointer[Float32, MutUntrackedOrigin],
    n_pixels_dp: Int64, actual_batch_dp: Int64,
    sample_clamp: Float32 = Float32(0.0),
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    # Ptex demand paging (ptex_mode != 0): redo_mask[sample] = 1 for the samples withheld here, to render again.
    redo_mask: Pointer[UInt8, MutUntrackedOrigin] = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
    ptex_mode: Int32 = Int32(0),
    aux_film: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    px_active: Pointer[UInt8, MutUntrackedOrigin] = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
    si_start: Int32 = Int32(0),
):
    var n_pixels = Int(n_pixels_dp)
    var actual_batch = Int(actual_batch_dp)
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    if px >= n_pixels:
        return
    var adaptive = _is_real_ptr(px_active)
    if adaptive and px_active[unsafe_offset=px] == UInt8(0) and si_start > Int32(0):
        # Converged: each skipped sample adds the pixel's mean, so the sums stay consistent with the global count.
        var k = Float32(actual_batch) / Float32(si_start)
        for c in range(3):
            film[unsafe_offset=px*3+c] += film[unsafe_offset=px*3+c] * k
            albedo_film[unsafe_offset=px*3+c] += albedo_film[unsafe_offset=px*3+c] * k
            aux_film[unsafe_offset=px*3+c] += aux_film[unsafe_offset=px*3+c] * k
        film_sq[unsafe_offset=px] += film_sq[unsafe_offset=px] * k
        return
    var xr = Float32(0); var xg = Float32(0); var xb = Float32(0)
    var r = Float32(0); var g = Float32(0); var b = Float32(0)
    var ar = Float32(0); var ag = Float32(0); var ab = Float32(0)
    var sq = Float32(0)
    for si in range(actual_batch):
        var p = paths[unsafe_offset=si * n_pixels + px]
        if ptex_mode != Int32(0):
            redo_mask[unsafe_offset=si * n_pixels + px] = UInt8(1) if p.ptex_missed == Int8(1) else UInt8(0)
            if p.ptex_missed != Int8(0):
                continue
        var _pe = spectral_sample_to_rgb(spectral_coeffs, Int(spectral_res_dp),
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
            p.estimate, p.wavelengths)
        var (_qr, _qg, _qb) = _clamp_sample_rgb(_pe[0], _pe[1], _pe[2], sample_clamp)
        r += _qr; g += _qg; b += _qb
        if (Int(si_start) + si) % 2 == 0:   # Cycles' class A samples, doubled
            xr += Float32(2) * _qr; xg += Float32(2) * _qg; xb += Float32(2) * _qb
        var _ql = RGB(_qr, _qg, _qb).luma()
        sq += _ql * _ql
        ar += p.albedo.r;  ag += p.albedo.g;  ab += p.albedo.b
    film[unsafe_offset=px*3+0] += r; film[unsafe_offset=px*3+1] += g; film[unsafe_offset=px*3+2] += b
    film_sq[unsafe_offset=px] += sq
    albedo_film[unsafe_offset=px*3+0] += ar; albedo_film[unsafe_offset=px*3+1] += ag; albedo_film[unsafe_offset=px*3+2] += ab
    if adaptive:
        aux_film[unsafe_offset=px*3+0] += xr; aux_film[unsafe_offset=px*3+1] += xg; aux_film[unsafe_offset=px*3+2] += xb


# Wavefront primary-ray generation: thread ti → pixel (ti % n_pixels), sample (si_start + ti // n_pixels).
# Layout: path_buf[si_local * n_pixels + px_flat] — adjacent threads touch adjacent pixels of same sample.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def gen_primary_rays_wavefront_gpu(
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    paths: Pointer[PathState, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    si_start: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count_dp: Int64, n_pixels_dp: Int64,
    filter_lut: Pointer[Float32, MutUntrackedOrigin],
    lens_radius: Float32, focal_distance: Float32,
    # Ptex demand paging, ptex_mode 2: slot ti renders sample redo_si[ti] of its pixel, or nothing if that is < 0.
    redo_si: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    ptex_mode: Int32 = Int32(0),
    px_active: Pointer[UInt8, MutUntrackedOrigin] = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),   # adaptive sampling
):
    var fw = Int(fw_dp)
    var count = Int(count_dp)
    var n_pixels = Int(n_pixels_dp)
    var ti = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ti >= count:
        return
    var skip = False
    var si_local = ti // n_pixels
    var px_flat  = ti % n_pixels
    var px = Int32(px_flat % fw)
    var py = Int32(px_flat // fw)
    var si = si_start + Int32(si_local)
    if ptex_mode == Int32(2):
        si = redo_si[unsafe_offset=ti]
        skip = si < Int32(0)
        si = max(si, Int32(0))
    if _is_real_ptr(px_active) and px_active[unsafe_offset=px_flat] == UInt8(0):
        skip = True
    var rng_seed = UInt64(rng_seed_hi) << UInt64(32) | UInt64(rng_seed_lo)
    var (ray, pcg_state, pcg_inc, sobol_idx, wavelengths) = gen_primary_ray_state(
        px, py, si, Int(log2spp), Int(n_base4),
        seed_dim0, seed_dim1, rng_seed, sobol_matrices, r2c, c2w,
        filter_norm_x, filter_sigma, filter_support_x,
        filter_norm_y, filter_support_y,
        filter_type, filter_lut,
        lens=ThinLens(lens_radius, focal_distance),
    )
    paths[unsafe_offset=ti] = PathState(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(0) if skip else Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Float32(1.0),   # previous_dielectric_ior (vacuum)
        Float32(1.0),   # eta_scale
        ThinLens(lens_radius, focal_distance).first_dim(), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
        INV_FOUR_PI,    # lastEnvNeePdf (gated by lastBsdfPdf > 0; set at each scatter)
        Float32(0.0),   # cone_len: total path length, accumulated per bounce
        Int8(2) if skip else Int8(0),   # ptex_missed
        MIN_RAY_PDF_NONE,   # min_ray_pdf
        SssWalk.none(),
    )


# Stream compaction (one atomic per block): idx[0..counter) lists the active paths, so later kernels run on full warps.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def compact_active_paths_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    count_dp: Int64,
    idx: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var act = tid < Int(count_dp) and paths[unsafe_offset=tid].active != 0
    var pos = warp_append(counter, act)
    if act:
        idx[unsafe_offset=pos] = Int32(tid)

# The same over an existing list, keeping the paths whose pending material is `kind`.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def compact_by_kind_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    src_idx: Pointer[Int32, MutUntrackedOrigin],
    src_count: Pointer[Int32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
    kind: Int32,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var p = 0
    var act = False
    if i < Int(src_count[unsafe_offset=0]):
        p = Int(src_idx[unsafe_offset=i])
        act = Int32(paths[unsafe_offset=p].pending_mat) == kind
    var pos = warp_append(counter, act)
    if act:
        idx[unsafe_offset=pos] = Int32(p)

# The paths on a subsurface walk, gathered so the walk kernel's warps hold only walking paths: spread among hair paths
# each warp waited for its longest walk.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def compact_walking_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    src_idx: Pointer[Int32, MutUntrackedOrigin],
    src_count: Pointer[Int32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var p = 0
    var act = False
    if i < Int(src_count[unsafe_offset=0]):
        p = Int(src_idx[unsafe_offset=i])
        act = paths[unsafe_offset=p].active != 0 and paths[unsafe_offset=p].sss.mesh >= Int32(0)
    var pos = warp_append(counter, act)
    if act:
        idx[unsafe_offset=pos] = Int32(p)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def sss_walk_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    idx: Pointer[Int32, MutUntrackedOrigin],
    count: Pointer[Int32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(count[unsafe_offset=0]):
        return
    _sss_walk_step(paths, intersections, Int(idx[unsafe_offset=i]), sd)


# One walk step for each listed walker (sss_walk_gpu's list); those still under the surface list themselves in next_idx
# with their next segment in `rays`, the RT-core layout, for rtcore_sss_walk_gpu to trace.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def sss_walk_rt_step_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    idx: Pointer[Int32, MutUntrackedOrigin],
    count: Pointer[Int32, MutUntrackedOrigin],
    next_idx: Pointer[Int32, MutUntrackedOrigin],
    next_count: Pointer[Int32, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var p = 0
    var pending = False
    if i < Int(count[unsafe_offset=0]):
        p = Int(idx[unsafe_offset=i])
        _sss_walk_step[1](paths, intersections, p, sd)
        pending = paths[unsafe_offset=p].active != 0 and paths[unsafe_offset=p].sss.mesh >= Int32(0)
    var pos = warp_append(next_count, pending)
    if pending:
        var ray = paths[unsafe_offset=p].ray
        var r = rays.unsafe_offset(8 * pos)
        r[unsafe_offset=0] = ray.origin.x; r[unsafe_offset=1] = ray.origin.y; r[unsafe_offset=2] = ray.origin.z
        r[unsafe_offset=3] = Float32(0)
        r[unsafe_offset=4] = ray.direction.x; r[unsafe_offset=5] = ray.direction.y; r[unsafe_offset=6] = ray.direction.z
        r[unsafe_offset=7] = Float32(1.0e8)
        next_idx[unsafe_offset=pos] = Int32(p)

# Adaptive sampling as Cycles (film_adaptive_sampling_convergence_check): a pixel stops when the difference between all
# its samples and twice every other one, over the square root of its intensity, falls under the threshold.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def adaptive_converge_gpu(
    film: Pointer[Float32, MutUntrackedOrigin],
    aux_film: Pointer[Float32, MutUntrackedOrigin],
    px_active: Pointer[UInt8, MutUntrackedOrigin],
    n_pixels_dp: Int64, spp: Int32, threshold: Float32,
):
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    if px >= Int(n_pixels_dp) or px_active[unsafe_offset=px] == UInt8(0):
        return
    var inv = Float32(1) / Float32(spp)
    var diff = Float32(0)
    var intensity = Float32(0)
    for c in range(3):
        var i = film[unsafe_offset=px*3+c] * inv
        diff += abs(i - aux_film[unsafe_offset=px*3+c] * inv)
        intensity += i
    var norm = sqrt(intensity) if intensity < Float32(1) else intensity
    if diff / (Float32(0.0001) + norm) < threshold:
        px_active[unsafe_offset=px] = UInt8(0)

# Cycles' film_adaptive_sampling_filter_x/_y: a pixel next to one still sampling samples too.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def adaptive_dilate_gpu(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[UInt8, MutUntrackedOrigin],
    width_dp: Int64, height_dp: Int64, vertical: Int32,
):
    var w = Int(width_dp)
    var h = Int(height_dp)
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    if px >= w * h:
        return
    var x = px % w
    var y = px // w
    var a = src[unsafe_offset=px]
    if vertical != Int32(0):
        if y > 0: a = max(a, src[unsafe_offset=px - w])
        if y < h - 1: a = max(a, src[unsafe_offset=px + w])
    else:
        if x > 0: a = max(a, src[unsafe_offset=px - 1])
        if x < w - 1: a = max(a, src[unsafe_offset=px + 1])
    dst[unsafe_offset=px] = a

# Traversal kernel that reads rays directly from PathState (no separate ray buffer).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_TRAVERSE))
def traverse_paths_gpu[rare_on: Bool, alpha_on: Bool, count_on: Bool = False](
    sd: SceneView,
    paths: Pointer[PathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    curve_cand_prim: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    count_dp: Int64,
    max_depth: Int32, fuse_cone: Int32,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
):
    var n_spheres = Int(sd.sphereCount)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    if paths[unsafe_offset=tid].active == 0:
        return
    # What deactivate_paths_past_maxdepth_gpu does, folded in: this kernel
    # already reads the path, and the separate pass was a full-grid launch.
    if paths[unsafe_offset=tid].bounce >= max_depth:
        paths[unsafe_offset=tid].at_cap = Int8(1)
    curve_cand_count[unsafe_offset=tid] = Int32(0)
    traverse_bvh2_core_defer_curves[rare_on, rare_on, alpha_on, count_on](
        sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, paths[unsafe_offset=tid].ray, Float32(1.0e38), results.unsafe_offset(tid),
        curve_cand_prim.unsafe_offset(tid * CURVE_DEFER_K), curve_cand_count.unsafe_offset(tid),
        sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
    )
    comptime if count_on:
        return
    test_spheres(sd.spheres, n_spheres, paths[unsafe_offset=tid].ray, results.unsafe_offset(tid))
    # accumulate_cone_gpu, folded in when no curve pass follows (that pass
    # rewrites tHit, so the cone must wait for it).
    if fuse_cone != Int32(0):
        _grow_cone(paths, results, tid)


# Task #163 stage 3: GPU-resident replacement for the `traverse_paths_gpu`
# dispatch in gpu_render_wavefront's per-bounce loop, routing the primary
# per-bounce intersection test through hardware ray tracing via the real
# CUDA/Vulkan interop mechanism (src/vulkaninterop -- zero CPU
# synchronization) instead of vulkanrt.mojo's host-round-trip
# vulkanrt_trace_rays (measured ~94x slower in an earlier version of this
# integration; see project_vulkan_rt_backend memory). Every other
# wavefront stage (medium sampling, all shade_*_gpu kernels, film
# accumulation) is completely unchanged -- they only ever read
# path_buf/inter_buf, never re-run primary-hit traversal themselves
# (shadow/NEE rays are a separate code path inside the shade kernels and
# stay on the CUDA/software BVH, not touched here).
#
# Pack ray data from path_buf directly into the interop-shared rays buffer
# -- no host copy, this kernel writes straight into CUDA-mapped memory
# that a Vulkan compute shader also reads.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vulkaninterop_pack_rays_kernel(
    paths: Pointer[PathState, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ray = paths[unsafe_offset=tid].ray
    var idx = tid * 8
    rays[unsafe_offset=idx + 0] = ray.origin.x
    rays[unsafe_offset=idx + 1] = ray.origin.y
    rays[unsafe_offset=idx + 2] = ray.origin.z
    rays[unsafe_offset=idx + 3] = Float32(1e-4)
    rays[unsafe_offset=idx + 4] = ray.direction.x
    rays[unsafe_offset=idx + 5] = ray.direction.y
    rays[unsafe_offset=idx + 6] = ray.direction.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8) if paths[unsafe_offset=tid].active != 0 else Float32(0.0)   # a finished path traces a zero-length ray

# Unpack the interop-shared results buffer (written by Vulkan's ray-query
# dispatch) directly into inter_buf's Intersection layout -- no host
# copy. Same materialIndex/area-light lookup and PrimId.type==3 encoding
# as the retired host-loop version (see finalize_scene in pbrt_parser.mojo
# for why area-light triangles need that special encoding, and
# project_vulkan_rt_backend memory for the real-bug story behind it) --
# now done per-thread on the GPU instead of a Mojo host loop, so
# mesh_material_idx/mesh_al_idx must be GPU-resident buffers here (see
# _build_mesh_light_info + the upload step in pipeline.mojo).
#
# Object instancing: a hit's hitMesh (instanceCustomIndex) is either an
# ordinary mesh index (< n_meshes, unchanged from before instancing existed)
# or mesh_count + instance_index (see vulkaninterop_rt_create_scene). For
# the latter, instance_base_mesh[instance_index] (host-precomputed as
# template_mesh_start[instances[k].blasIdx], pipeline.mojo) plus the hit's
# geometryIndex gives the real originating mesh index -- AreaLightSource is
# not supported inside ObjectBegin/ObjectEnd (pbrt_parser.mojo skips it
# there), so instance hits are always ordinary (type==0) triangles, never
# the type==3 area-light encoding.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vulkaninterop_unpack_results_kernel(
    results: Pointer[Float32, MutUntrackedOrigin],
    inter: Pointer[Intersection, MutUntrackedOrigin],
    mesh_material_idx: Pointer[Int64, MutUntrackedOrigin],
    mesh_al_idx: Pointer[Int32, MutUntrackedOrigin],
    n_meshes_dp: Int64,
    count_dp: Int64,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    # --rt-hardware: `results` holds only the listed (live) rays; ids maps slot -> path, and paths (if real) gets the cone growth fused in.
    ids: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    paths: Pointer[PathState, MutUntrackedOrigin] = Pointer[PathState, MutUntrackedOrigin].unsafe_dangling(),
    # A merged template mesh keeps a material a triangle in the mesh itself; real pointer = look it up there.
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin] = Pointer[TriangleMesh, MutUntrackedOrigin].unsafe_dangling(),
):
    var n_meshes = Int(n_meshes_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid * 8
    var ot = Int(ids[unsafe_offset=tid]) if _is_real_ptr(ids) else tid
    var iresults = results.unsafe_bitcast[Int32]()
    var hitFlag = iresults[unsafe_offset=idx + 6]
    if hitFlag == Int32(1):
        var raw_idx = Int(iresults[unsafe_offset=idx + 4])
        var tri = iresults[unsafe_offset=idx + 5]
        var geometry_idx = iresults[unsafe_offset=idx + 7]
        var instance_idx = Int32(-1)
        var mi = raw_idx
        if raw_idx >= n_meshes and _is_real_ptr(instance_base_mesh):
            instance_idx = Int32(raw_idx - n_meshes)
            mi = Int(instance_base_mesh[unsafe_offset=Int(instance_idx)]) + Int(geometry_idx)
        var mat_idx = Int64(0)
        var al = Int32(-1)
        if mi >= 0 and mi < n_meshes:
            mat_idx = mesh_material_idx[unsafe_offset=mi]
            if instance_idx < Int32(0):
                al = mesh_al_idx[unsafe_offset=mi]
            if _is_real_ptr(meshes) and _is_real_ptr(meshes[unsafe_offset=mi].materials):
                mat_idx = Int64(mesh_material(meshes[unsafe_offset=mi], Int(tri)))
        var hitT = results[unsafe_offset=idx + 0]
        var u = results[unsafe_offset=idx + 1]
        var v = results[unsafe_offset=idx + 2]
        if al >= Int32(0):
            inter[unsafe_offset=ot] = Intersection(
                PrimId(Int64(al), (Int64(mi) << 32) | Int64(tri), mat_idx, Int32(-1),
                         Int8(3), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
        else:
            inter[unsafe_offset=ot] = Intersection(
                PrimId(Int64(mi), Int64(tri) * 3, mat_idx, instance_idx,
                         Int8(0), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
    elif hitFlag == Int32(2) or hitFlag == Int32(4):
        # Curve hit, resolved directly by intersect_batch.comp itself (real
        # ray-vs-curve narrow-phase test + rayQueryGenerateIntersectionEXT
        # on a valid hit) -- no candidate buffer, no separate CUDA resolve
        # pass. hitMesh/hitTriangle/geometryIndex already carry PrimId's
        # id1 (curve index)/id2 (packed piece info)/materialIndex directly
        # (see vulkaninterop_rt_create_scene's docstring), so this is a
        # straight repack, no lookups needed. u/v here are intersect_curve's
        # own (h, v) outputs, not barycentrics.
        var curve_idx = Int64(iresults[unsafe_offset=idx + 4])
        var piece_info = Int64(iresults[unsafe_offset=idx + 5])
        var mat_idx = Int64(iresults[unsafe_offset=idx + 7])
        var curve_instance = iresults[unsafe_offset=idx + 3] if hitFlag == Int32(4) else Int32(-1)   # 4: the RT-core hybrid, curve of an instance
        var hitT = results[unsafe_offset=idx + 0]
        var h = results[unsafe_offset=idx + 1]
        var v = results[unsafe_offset=idx + 2]
        inter[unsafe_offset=ot] = Intersection(
            PrimId(curve_idx, piece_info, mat_idx, curve_instance, Int8(5), Int8(0), Int8(0), Int8(0)),
            hitT, h, v, Int8(1), Int8(0), Int8(0), Int8(0),
        )
    else:
        # tHit must be reset to the "no hit yet" sentinel (matching
        # traverse_bvh2_core_defer_curves's convention), not just hit=0 --
        # vulkaninterop_test_spheres_gpu (called right after this kernel)
        # uses result[0].tHit as its starting max-distance. Leaving it at
        # whatever inter_buf held from an earlier bounce would silently
        # reject a real closer sphere hit as "farther than the (stale)
        # current best".
        var dummy_id = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
        inter[unsafe_offset=ot] = Intersection(dummy_id, Float32(1.0e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    if _is_real_ptr(paths) and hitFlag != Int32(3):
        _grow_cone(paths, inter, ot)

# Analytic sphere test as a SEPARATE pass after the Vulkan-RT-traced mesh/
# instance hit above -- exactly mirrors how traverse_paths_gpu (the pure-
# CUDA path) composes traverse_bvh2_core_defer_curves + test_spheres
# already: test_spheres itself keeps whichever hit is closer (it reads
# result[0].tHit as its starting max-distance when result[0].hit != 0), so
# calling it here with whatever vulkaninterop_unpack_results_kernel just
# wrote is correct without any extra "closest of two" logic on this side.
# Spheres stay purely analytic (exact intersection/normals) rather than
# tessellated into the Vulkan BLAS/TLAS -- simpler, and free of any
# tessellation-precision tradeoff.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vulkaninterop_test_spheres_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    inter: Pointer[Intersection, MutUntrackedOrigin],
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    count_dp: Int64,
):
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[unsafe_offset=tid].active == 0:
        return
    test_spheres(spheres, n_spheres, paths[unsafe_offset=tid].ray, inter.unsafe_offset(tid))

# Host driver: enqueues pack -> interop ray-query dispatch -> unpack, ALL
# on ctx.stream() in strict program order, with NO ctx.synchronize()
# anywhere -- vulkaninterop_rt_trace's internal CUDA-signal/Vulkan-submit/
# CUDA-wait handoff (see vulkaninterop.h) is what keeps the Vulkan
# dispatch correctly ordered relative to the pack/unpack kernels on either
# side of it, entirely on the GPU timeline. This is what makes stage 3
# fast where the retired host-round-trip version was ~94x slower: no CPU
# stall, no host-memory copy, anywhere in this function.
#
# Scope: triangle/instance/curve geometry via Vulkan RT, PLUS an analytic
# sphere pass (see vulkaninterop_test_spheres_gpu) -- meshes, instances,
# spheres, AND curves are all supported now (see vulkaninterop_rt_create_
# scene's docstring for how curves are represented).
@always_inline
def _grow_cone(
    paths: Pointer[PathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    tid: Int,
):
    if results[unsafe_offset=tid].hit == Int8(0):
        return
    # ONLY along a specular chain. A ray cone tracks the CAMERA's footprint,
    # and that survives mirror reflection and refraction -- but at a diffuse
    # scatter the outgoing direction is random and the cone stops meaning
    # anything about the camera. Growing it there would blur deeper bounces
    # without bound and without justification; pbrt carries differentials for
    # camera rays and specular chains for the same reason. After the first
    # non-specular scatter pbrt's ray carries no differentials at all and every
    # later hit takes Camera::Approximate_dp_dxy; cone_len = -1 records that.
    if paths[unsafe_offset=tid].cone_len < Float32(0.0):
        return
    if paths[unsafe_offset=tid].bounce == Int32(0) or paths[unsafe_offset=tid].specularBounce != Int8(0):
        paths[unsafe_offset=tid].cone_len += results[unsafe_offset=tid].tHit
    else:
        paths[unsafe_offset=tid].cone_len = Float32(-1.0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def accumulate_cone_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Grow each path's texture-footprint cone by the segment just traced.

    Its own kernel, enqueued after WHICHEVER traversal ran, because there are
    two (the CUDA `traverse_paths_gpu` and the Vulkan-RT one, whose unpack
    kernel never sees `paths`). Putting it inside a material's shading instead
    would skip exactly the paths that matter: a camera ray refracting through
    water onto a textured floor never touches the diffuse material's geometry
    builder on the way through the water.
    """
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[unsafe_offset=tid].active == 0:
        return
    _grow_cone(paths, results, tid)



# ── Alpha cutouts on the RT cores ──────────────────────────────────────────────
# The hardware geometry is opaque, so a ray returns the nearest triangle even when its alpha cutout rejects it. The
# callers trace once, then this pass applies alpha_killed (the same test the software BVH runs) to every ray's hit: a
# rejected hit moves the ray's t_min just past it and the ray goes on a compact list. Only the (few) listed rays are
# gathered, re-traced and re-tested, up to RT_ALPHA_PASSES times; a ray whose hit survives has its final result copied
# back into `results`. Rays still listed after the last pass get hit flag 3 (dense foliage) so the software BVH, which has
# no depth limit, resolves them -- see rtcore_alpha_fallback_gpu and the shadow resolve kernel.
# Scratch layout (N = scratch bytes / 32 rays): compact results [0, 8N), compact rays [8N, 16N), list A [16N, 20N),
# list B [20N, 24N), two counters [24N, 24N + 8).
comptime RT_ALPHA_PASSES = 32

@always_inline
def _alpha_rejects(
    rays: Pointer[Float32, MutUntrackedOrigin],
    ridx: Int,
    src: Pointer[Float32, MutUntrackedOrigin],
    sidx: Int,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    n_meshes: Int,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin],
) -> Bool:
    var isrc = src.unsafe_bitcast[Int32]()
    if isrc[unsafe_offset=sidx + 6] != Int32(1):
        return False
    var raw_idx = Int(isrc[unsafe_offset=sidx + 4])
    var tri = Int(isrc[unsafe_offset=sidx + 5])
    var mi = raw_idx
    if raw_idx >= n_meshes and _is_real_ptr(instance_base_mesh):
        mi = Int(instance_base_mesh[unsafe_offset=raw_idx - n_meshes]) + Int(isrc[unsafe_offset=sidx + 7])
    if mi < 0 or mi >= n_meshes:
        return False
    var mesh = meshes[unsafe_offset=mi]
    if mesh.alpha_w == Int32(0) and mesh.alpha_const >= Float32(1.0):
        return False
    var v0 = Int(mesh_vi(mesh, tri * 3))
    var v1 = Int(mesh_vi(mesh, tri * 3 + 1))
    var v2 = Int(mesh_vi(mesh, tri * 3 + 2))
    var org = Vec3f(rays[unsafe_offset=ridx], rays[unsafe_offset=ridx + 1], rays[unsafe_offset=ridx + 2])
    var dir = Vec3f(rays[unsafe_offset=ridx + 4], rays[unsafe_offset=ridx + 5], rays[unsafe_offset=ridx + 6])
    return alpha_killed(mesh, v0, v1, v2, src[unsafe_offset=sidx + 1], src[unsafe_offset=sidx + 2], org, dir, (mi << 32) | (tri * 3))

# Appends to a list through ONE global atomic per block of 256 threads (a same-address atomic costs ~15 ns, so one per warp
# was already the bottleneck): every thread of the block must call this (no early return before it). Returns -1 if !live.
@always_inline
def warp_append(counter: Pointer[Int32, MutUntrackedOrigin], live: Bool) -> Int:
    var warp_counts = stack_allocation[8, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var block_base = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    comptime MaskT = DType.uint32 if WARP_SIZE == 32 else DType.uint64
    var mask = vote[MaskT](live)
    var lane = lane_id()
    var warp = Int(thread_idx.x) // WARP_SIZE
    if lane == 0:
        warp_counts[unsafe_offset=warp] = Int32(pop_count(mask))
    barrier()
    if thread_idx.x == 0:
        var total = Int32(0)
        for w in range(256 // WARP_SIZE):
            var c = warp_counts[unsafe_offset=w]
            warp_counts[unsafe_offset=w] = total
            total += c
        block_base[unsafe_offset=0] = Atomic.fetch_add(counter, total) if total > Int32(0) else Int32(0)
    barrier()
    if not live:
        return -1
    return Int(block_base[unsafe_offset=0]) + Int(warp_counts[unsafe_offset=warp]) + Int(pop_count(mask & ((Scalar[MaskT](1) << Scalar[MaskT](lane)) - Scalar[MaskT](1))))

def rt_no_ids() -> Pointer[Int32, MutUntrackedOrigin]:
    return Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()

@always_inline
def rt_no_paths() -> Pointer[PathState, MutUntrackedOrigin]:
    return Pointer[PathState, MutUntrackedOrigin].unsafe_dangling()

@always_inline
def rt_no_meshes() -> Pointer[TriangleMesh, MutUntrackedOrigin]:
    return Pointer[TriangleMesh, MutUntrackedOrigin].unsafe_dangling()

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_alpha_first_kernel(
    rays: Pointer[Float32, MutUntrackedOrigin],
    results: Pointer[Float32, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    n_meshes_dp: Int64,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin],
    count_dp: Int64,
    list_out: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var idx = tid * 8
    var killed = False
    if tid < Int(count_dp):
        if rays[unsafe_offset=idx + 7] > Float32(0.0):    # a finished path / empty shadow slot is a zero-length ray
            killed = _alpha_rejects(rays, idx, results, idx, meshes, Int(n_meshes_dp), instance_base_mesh)
            if killed:
                rays[unsafe_offset=idx + 3] = results[unsafe_offset=idx] * Float32(1.00001) + Float32(1.0e-5)
    var pos = warp_append(counter, killed)
    if killed:
        list_out[unsafe_offset=pos] = Int32(tid)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_alpha_gather_kernel(
    rays: Pointer[Float32, MutUntrackedOrigin],
    rays_c: Pointer[Float32, MutUntrackedOrigin],
    list_in: Pointer[Int32, MutUntrackedOrigin],
    m_dp: Int64,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(m_dp):
        return
    var src = Int(list_in[unsafe_offset=i]) * 8
    for k in range(8):
        rays_c[unsafe_offset=i * 8 + k] = rays[unsafe_offset=src + k]

# Re-test the re-traced rays: a rejected hit stays listed (into list_out), anything else is final and copied back.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_alpha_retest_kernel(
    rays: Pointer[Float32, MutUntrackedOrigin],
    rays_c: Pointer[Float32, MutUntrackedOrigin],
    results: Pointer[Float32, MutUntrackedOrigin],
    results_c: Pointer[Float32, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    n_meshes_dp: Int64,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin],
    list_in: Pointer[Int32, MutUntrackedOrigin],
    m_dp: Int64,
    list_out: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var killed = False
    var dst: Int
    if i < Int(m_dp):
        dst = Int(list_in[unsafe_offset=i]) * 8
        killed = _alpha_rejects(rays_c, i * 8, results_c, i * 8, meshes, Int(n_meshes_dp), instance_base_mesh)
        if killed:
            rays[unsafe_offset=dst + 3] = results_c[unsafe_offset=i * 8] * Float32(1.00001) + Float32(1.0e-5)
        else:
            for k in range(8):
                results[unsafe_offset=dst + k] = results_c[unsafe_offset=i * 8 + k]
    var pos = warp_append(counter, killed)
    if killed:
        list_out[unsafe_offset=pos] = list_in[unsafe_offset=i]

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_alpha_finish_kernel(
    results: Pointer[Float32, MutUntrackedOrigin],
    list_in: Pointer[Int32, MutUntrackedOrigin],
    m_dp: Int64,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(m_dp):
        return
    results.unsafe_bitcast[Int32]()[unsafe_offset=Int(list_in[unsafe_offset=i]) * 8 + 6] = Int32(3)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_zero_counters_kernel(counters: Pointer[Int32, MutUntrackedOrigin]):
    if thread_idx.x == 0:
        counters[unsafe_offset=0] = Int32(0)
        counters[unsafe_offset=1] = Int32(0)

# Reads a small device counter buffer back (one pageable 16-byte copy; this synchronizes the stream).
def read_counter(ctx: DeviceContext, counter_buf: DeviceBuffer[DType.uint8], index: Int = 0) raises -> Int:
    var host = unsafe_alloc[UInt8](len(counter_buf))
    ctx.enqueue_copy(host, counter_buf)
    ctx.synchronize()
    var v = Int(host.unsafe_bitcast[Int32]()[unsafe_offset=index])
    host.unsafe_free()
    return v

# Called after the first RT-core trace into `results` (interop layout). Does nothing for scenes without alpha.
def rtcore_alpha_passes(
    ctx: DeviceContext,
    rays_buf: DeviceBuffer[DType.float32],
    results_buf: DeviceBuffer[DType.float32],
    scratch_buf: DeviceBuffer[DType.uint8],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    n_meshes: Int,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin],
    n_total: Int,
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)
    var cuda_stream = CUDA(ctx.stream())
    var rt_hw = rtcore_active()
    var n_cap = len(scratch_buf) // 32
    var chunk = max(n_cap // 4, 1)
    var base = scratch_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var results_c = base.unsafe_bitcast[Float32]()
    var rays_c = base.unsafe_offset(8 * n_cap).unsafe_bitcast[Float32]()
    var list_a = base.unsafe_offset(16 * n_cap).unsafe_bitcast[Int32]()
    var list_b = base.unsafe_offset(20 * n_cap).unsafe_bitcast[Int32]()
    var counters = base.unsafe_offset(24 * n_cap).unsafe_bitcast[Int32]()
    var counter_buf = scratch_buf.create_sub_buffer[DType.uint8](24 * n_cap, 8)
    var rays = rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var results = results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ctx.enqueue_function[rtcore_zero_counters_kernel](counters, grid_dim=1, block_dim=32)
    ctx.enqueue_function[rtcore_alpha_first_kernel](
        rays, results, meshes, Int64(n_meshes), instance_base_mesh, Int64(n_total), list_a, counters,
        grid_dim=grid, block_dim=block_size,
    )
    var cnt = read_counter(ctx, counter_buf, 0)
    var list_in = list_a
    var list_out = list_b
    var p = 0
    while cnt > 0 and (p + 1) < RT_ALPHA_PASSES:
        p += 1
        ctx.enqueue_function[rtcore_zero_counters_kernel](counters, grid_dim=1, block_dim=32)
        var off = 0
        while off < cnt:
            var m = min(chunk, cnt - off)
            var g = ceildiv(m, block_size)
            ctx.enqueue_function[rtcore_alpha_gather_kernel](
                rays, rays_c, list_in.unsafe_offset(off), Int64(m), grid_dim=g, block_dim=block_size,
            )
            _ = rtcore_trace_interop(rt_hw, UInt64(Int(rays_c)), UInt64(Int(results_c)), Int32(m), cuda_stream)
            ctx.enqueue_function[rtcore_alpha_retest_kernel](
                rays, rays_c, results, results_c, meshes, Int64(n_meshes), instance_base_mesh, list_in.unsafe_offset(off), Int64(m),
                list_out, counters,
                grid_dim=g, block_dim=block_size,
            )
            off += m
        cnt = read_counter(ctx, counter_buf, 0)
        var tmp = list_in
        list_in = list_out
        list_out = tmp
    if cnt > 0 and getenv("GONZALES_RT_ALPHA_LOG") != "":
        print("alpha leftover", cnt, "of", n_total, "after", p + 1, "passes")
    if cnt > 0:
        ctx.enqueue_function[rtcore_alpha_finish_kernel](
            results, list_in, Int64(cnt), grid_dim=ceildiv(cnt, block_size), block_dim=block_size,
        )


# Reads the ray from the interop rays buffer (so it serves any state layout: path tracer, VCM light and camera paths).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_alpha_fallback_gpu(
    sd: SceneView,
    rays: Pointer[Float32, MutUntrackedOrigin],
    interop_results: Pointer[Float32, MutUntrackedOrigin],
    inter: Pointer[Intersection, MutUntrackedOrigin],
    count_dp: Int64,
    ids: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    paths: Pointer[PathState, MutUntrackedOrigin] = Pointer[PathState, MutUntrackedOrigin].unsafe_dangling(),
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if interop_results.unsafe_bitcast[Int32]()[unsafe_offset=tid * 8 + 6] != Int32(3):
        return
    var idx = tid * 8
    var ot = Int(ids[unsafe_offset=tid]) if _is_real_ptr(ids) else tid
    var ray = Ray(Point3f(rays[unsafe_offset=idx], rays[unsafe_offset=idx + 1], rays[unsafe_offset=idx + 2]),
                  Vec3f(rays[unsafe_offset=idx + 4], rays[unsafe_offset=idx + 5], rays[unsafe_offset=idx + 6]))
    # Alpha scenes are triangle-only here (curves are not eligible), so the curve candidate buffers are never touched.
    traverse_bvh2_core_defer_curves(
        sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), inter.unsafe_offset(ot),
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
        sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
    )
    if _is_real_ptr(paths):
        _grow_cone(paths, inter, ot)

# Analytic spheres for rays that live in the interop rays buffer (VCM's path states are not PathState).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_spheres_from_rays_gpu(
    rays: Pointer[Float32, MutUntrackedOrigin],
    inter: Pointer[Intersection, MutUntrackedOrigin],
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid * 8
    var ray = Ray(Point3f(rays[unsafe_offset=idx], rays[unsafe_offset=idx + 1], rays[unsafe_offset=idx + 2]),
                  Vec3f(rays[unsafe_offset=idx + 4], rays[unsafe_offset=idx + 5], rays[unsafe_offset=idx + 6]))
    test_spheres(spheres, Int(n_spheres_dp), ray, inter.unsafe_offset(tid))

comptime CURVE_BLOCK: Int = get_defined_int["CURVE_BLOCK", 32]()


@always_inline
def _curves_apart(sd: SceneView) -> Bool:
    """The curves are in the clusters only, not in the scene BVH (ParsedScene_Mojo.curve_first)."""
    return sd.curveBvh.n_clusters > Int32(0) and sd.curveRoot < Int32(0)


# The RT cores trace triangles only: curves are tested in software against the RT hit distance, and a nearer curve hit
# replaces the raw result (flag 2, or 4 with the instance in word 3) so the unpack kernel treats it like any curve hit.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rtcore_curve_clip_gpu[clusters: Bool](
    sd: SceneView,
    rays: Pointer[Float32, MutUntrackedOrigin],
    results: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(count_dp):
        return
    var idx = tid * 8
    var ir = results.unsafe_bitcast[Int32]()
    var flag = ir[unsafe_offset=idx + 6]
    if flag == Int32(3):
        return
    var tmax = rays[unsafe_offset=idx + 7]
    if flag == Int32(1):
        tmax = results[unsafe_offset=idx + 0]
    var ray = Ray(Point3f(rays[unsafe_offset=idx], rays[unsafe_offset=idx + 1], rays[unsafe_offset=idx + 2]),
                  Vec3f(rays[unsafe_offset=idx + 4], rays[unsafe_offset=idx + 5], rays[unsafe_offset=idx + 6]))
    var tmp = Intersection(PrimId(-1, -1, 0, -1, 0, 0, 0, 0), tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    comptime if clusters:
        var h = curve_bvh_closest(sd.curveBvh, sd.curves, ray, tmax)
        if h[0] < 0:
            return
        tmp = Intersection(curve_bvh_prim(sd.curveBvh, h[0]), h[1], h[2], h[3], Int8(1), Int8(0), Int8(0), Int8(0))
    else:
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, tmax, Pointer(to=tmp),
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, root=max(Int(sd.curveRoot), 0))
    if tmp.hit == Int8(0) or tmp.primId.type != Int8(5):
        return
    results[unsafe_offset=idx + 0] = tmp.tHit
    results[unsafe_offset=idx + 1] = tmp.u
    results[unsafe_offset=idx + 2] = tmp.v
    ir[unsafe_offset=idx + 3] = tmp.primId.instanceIdx
    ir[unsafe_offset=idx + 4] = Int32(tmp.primId.id1)
    ir[unsafe_offset=idx + 5] = Int32(tmp.primId.id2)
    ir[unsafe_offset=idx + 7] = Int32(tmp.primId.materialIndex)
    ir[unsafe_offset=idx + 6] = Int32(4) if tmp.primId.instanceIdx >= Int32(0) else Int32(2)

# The RT-core replacement for "trace + unpack": traces the interop rays buffer on the RT cores into the interop result layout,
# re-traces past rejected alpha hits (scenes with alpha only), decodes into Intersections, and lets the software BVH resolve
# rays that were rejected too often. Spheres are the caller's business (they need the caller's ray layout).
def rtcore_trace_unpack_gpu(
    ctx: DeviceContext,
    inter_buf: DeviceBuffer[DType.uint8],
    sd: SceneView,
    rays_buf: DeviceBuffer[DType.float32],
    results_buf: DeviceBuffer[DType.float32],
    scratch_buf: DeviceBuffer[DType.uint8],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]],
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)
    var cuda_stream = CUDA(ctx.stream())
    var rt_hw = rtcore_active()
    _ = rtcore_trace_interop(rt_hw, UInt64(Int(rays_buf.unsafe_ptr())), UInt64(Int(results_buf.unsafe_ptr())), Int32(n_total), cuda_stream)
    var instance_base_mesh_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
    if instance_base_mesh_buf:
        instance_base_mesh_ptr = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var alpha = Int(rtcore_alpha_enabled()) != 0
    if alpha:
        rtcore_alpha_passes(ctx, rays_buf, results_buf, scratch_buf, sd.meshes, n_meshes, instance_base_mesh_ptr, n_total)
    if Int(sd.curveCount) > 0:
        var rp = rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        var sp = results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        var ox = oxc_active()
        if Int(ox) != 0:
            _ = oxc_trace_closest(ox, UInt64(Int(rp)), UInt64(Int(sp)), n_total, CUDA(ctx.stream()))
        elif sd.curveBvh.n_clusters > Int32(0):
            ctx.enqueue_function[rtcore_curve_clip_gpu[True]](sd, rp, sp, Int64(n_total), grid_dim=grid, block_dim=block_size)
        else:
            ctx.enqueue_function[rtcore_curve_clip_gpu[False]](sd, rp, sp, Int64(n_total), grid_dim=grid, block_dim=block_size)
    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_meshes),
        Int64(n_total),
        instance_base_mesh_ptr,
        rt_no_ids(), rt_no_paths(), sd.meshes,
        grid_dim=grid, block_dim=block_size,
    )
    if alpha:
        ctx.enqueue_function[rtcore_alpha_fallback_gpu](
            sd,
            rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(n_total), rt_no_ids(), rt_no_paths(),
            grid_dim=grid, block_dim=block_size,
        )


# ── Compacted RT-core passes (PT) ─────────────────────────────────────────────
# Most paths are dead after a few bounces and most shadow slots are empty, yet every full-grid pack/trace/convert/unpack
# pass touched all n_pixels * WAVEFRONT_BATCH slots. These kernels list only the live rays (one atomic per warp) so every
# later pass runs over a short dense array; the host reads the count back once per stage.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rt_pack_compact_kernel(
    paths: Pointer[PathState, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    ids: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
    count_dp: Int64,
    max_depth: Int32,
    src_ids: Pointer[Int32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var tid = i
    var live = False
    if i < Int(count_dp):
        if _is_real_ptr(src_ids):                          # paths only ever die: last round's live paths are the candidates
            tid = Int(src_ids[unsafe_offset=i])
        live = paths[unsafe_offset=tid].active != Int8(0)
        if live and paths[unsafe_offset=tid].bounce >= max_depth:
            paths[unsafe_offset=tid].at_cap = Int8(1)      # see deactivate_paths_past_maxdepth_gpu
    var pos = warp_append(counter, live)
    if not live:
        return
    var ray = paths[unsafe_offset=tid].ray
    var idx = pos * 8
    rays[unsafe_offset=idx + 0] = ray.origin.x
    rays[unsafe_offset=idx + 1] = ray.origin.y
    rays[unsafe_offset=idx + 2] = ray.origin.z
    rays[unsafe_offset=idx + 3] = Float32(0)          # the origin is already offset, as for the software traversal (t > 0)
    rays[unsafe_offset=idx + 4] = ray.direction.x
    rays[unsafe_offset=idx + 5] = ray.direction.y
    rays[unsafe_offset=idx + 6] = ray.direction.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8)
    ids[unsafe_offset=pos] = Int32(tid)

# GONZALES_RAY_DUMP=<prefix>: writes the packed rays of one frame and times the bare RT-core trace of them.
def _ray_dump(ctx: DeviceContext, rays_buf: DeviceBuffer[DType.float32], results_buf: DeviceBuffer[DType.float32], m: Int, tag: String) raises:
    var host = unsafe_alloc[Float32](len(rays_buf))
    ctx.enqueue_copy(host, rays_buf)
    ctx.synchronize()
    var path = getenv("GONZALES_RAY_DUMP") + "_" + tag + ".bin"
    var c = unsafe_alloc[UInt8](path.byte_length() + 3)
    for k in range(path.byte_length()): c[unsafe_offset=k] = path.unsafe_ptr()[unsafe_offset=k]
    c[unsafe_offset=path.byte_length()] = UInt8(0)
    c[unsafe_offset=path.byte_length() + 1] = UInt8(119); c[unsafe_offset=path.byte_length() + 2] = UInt8(0)
    var f = external_call["fopen", Int, Pointer[UInt8, MutUntrackedOrigin], Pointer[UInt8, MutUntrackedOrigin]](c, c.unsafe_offset(path.byte_length() + 1))
    if f != 0:
        _ = external_call["fwrite", Int, Pointer[Float32, MutUntrackedOrigin], Int, Int, Int](host, 32, m, f)
        _ = external_call["fclose", Int32, Int](f)
    c.unsafe_free()
    var rt = rtcore_active()
    _ = rtcore_trace_interop(rt, UInt64(Int(rays_buf.unsafe_ptr())), UInt64(Int(results_buf.unsafe_ptr())), Int32(m), CUDA(ctx.stream()))
    ctx.enqueue_copy(host, results_buf)
    ctx.synchronize()
    var nh = 0
    var tsum = 0.0
    for k in range(m):
        if host.unsafe_bitcast[Int32]()[unsafe_offset=k * 8 + 6] == Int32(1):
            nh += 1
            tsum += Float64(host[unsafe_offset=k * 8])
    host.unsafe_free()
    print("raydump", tag, ":", nh, "hits, mean t", tsum / Float64(max(nh, 1)))
    var best = 1.0e30
    var total = 0.0
    for k in range(23):
        var t0 = perf_counter_ns()
        _ = rtcore_trace(rt, UInt64(Int(rays_buf.unsafe_ptr())), UInt64(Int(results_buf.unsafe_ptr())), Int32(m), CUDA(ctx.stream()))
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) / 1.0e6
        if k >= 3:
            total += ms
            best = min(best, ms)
    print("raydump", tag, ":", m, "rays, bare trace mean", total / 20.0, "ms (", Float64(m) / total * 20.0 / 1000.0, "Mray/s ), best", best, "ms")

# Closest hit of every live path on the RT cores; writes the Intersections of the listed paths only (a dead path's
# Intersection is never read: every later kernel checks PathState.active first).
def rtcore_closest_hit_gpu(
    ctx: DeviceContext,
    path_buf: DeviceBuffer[DType.uint8],
    inter_buf: DeviceBuffer[DType.uint8],
    sd: SceneView,
    rays_buf: DeviceBuffer[DType.float32],
    results_buf: DeviceBuffer[DType.float32],
    scratch_buf: DeviceBuffer[DType.uint8],
    ids_buf: DeviceBuffer[DType.uint8],
    counter_buf: DeviceBuffer[DType.uint8],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
    max_depth: Int32,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]],
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    n_spheres: Int,
    round: Int,
    prev_live: Int,
    bench: Pointer[Int, MutUntrackedOrigin],
) raises -> Int:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)
    var cuda_stream = CUDA(ctx.stream())
    var rt_hw = rtcore_active()
    var paths = path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var rays = rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var results = results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var inter = inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    # ids_buf holds three lists of n_total entries: live paths of even rounds, of odd rounds, and the shadow tasks.
    var ids_base = ids_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var ids = ids_base.unsafe_offset((round % 2) * n_total)
    var src_ids = ids_base.unsafe_offset(((round + 1) % 2) * n_total) if round > 0 else rt_no_ids()
    var n_scan = prev_live if round > 0 else n_total
    ctx.enqueue_memset(counter_buf, UInt8(0))
    if n_scan > 0:
        ctx.enqueue_function[rt_pack_compact_kernel](
            paths, rays, ids, counter_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), Int64(n_scan), max_depth, src_ids,
            grid_dim=ceildiv(n_scan, block_size), block_dim=block_size,
        )
    var m = read_counter(ctx, counter_buf)
    if m == 0:
        return 0
    bench[unsafe_offset=0] += m
    if bench[unsafe_offset=7] != 0:
        _ray_dump(ctx, rays_buf, results_buf, m, "c" + String(round))
    var t0 = perf_counter_ns()
    var gm = ceildiv(m, block_size)
    var instance_base_mesh_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
    if instance_base_mesh_buf:
        instance_base_mesh_ptr = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    _ = rtcore_trace_interop(rt_hw, UInt64(Int(rays_buf.unsafe_ptr())), UInt64(Int(results_buf.unsafe_ptr())), Int32(m), cuda_stream)
    var alpha = Int(rtcore_alpha_enabled()) != 0
    if alpha:
        rtcore_alpha_passes(ctx, rays_buf, results_buf, scratch_buf, sd.meshes, n_meshes, instance_base_mesh_ptr, m)
    if Int(sd.curveCount) > 0:
        var ox = oxc_active()
        if Int(ox) != 0:
            _ = oxc_trace_closest(ox, UInt64(Int(rays)), UInt64(Int(results)), m, CUDA(ctx.stream()))
        elif sd.curveBvh.n_clusters > Int32(0):
            ctx.enqueue_function[rtcore_curve_clip_gpu[True]](sd, rays, results, Int64(m), grid_dim=ceildiv(m, CURVE_BLOCK), block_dim=CURVE_BLOCK)
        else:
            ctx.enqueue_function[rtcore_curve_clip_gpu[False]](sd, rays, results, Int64(m), grid_dim=ceildiv(m, CURVE_BLOCK), block_dim=CURVE_BLOCK)
    var fuse_paths = paths if n_spheres == 0 else Pointer[PathState, MutUntrackedOrigin].unsafe_dangling()
    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        results, inter,
        mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_meshes), Int64(m), instance_base_mesh_ptr, ids, fuse_paths, sd.meshes,
        grid_dim=gm, block_dim=block_size,
    )
    if alpha:
        ctx.enqueue_function[rtcore_alpha_fallback_gpu](
            sd, rays, results, inter, Int64(m), ids, fuse_paths,
            grid_dim=gm, block_dim=block_size,
        )
    if n_spheres > 0:
        ctx.enqueue_function[vulkaninterop_test_spheres_gpu](
            paths, inter, spheres, Int64(n_spheres), Int64(n_total),
            grid_dim=grid, block_dim=block_size,
        )
        ctx.enqueue_function[accumulate_cone_gpu](paths, inter, Int64(n_total), grid_dim=grid, block_dim=block_size)
    if bench[unsafe_offset=3] == 2:
        ctx.synchronize()
        bench[unsafe_offset=2] += Int(perf_counter_ns() - t0)
    return m


# Software deferred shadow rays: the shade kernels only queue ShadowTasks; these two kernels trace them with the software
# any-hit. The first lists the live paths that queued at least one task (one atomic per block), the second traces a listed
# path's tasks in slot order (one thread per path, so the estimate is never updated concurrently) and frees them.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
# One thread per (live path, slot): lists every queued shadow ray, so a path's two rays are traced by two threads.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def sw_pack_shadow_tasks_gpu(
    tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    ids: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var t = 0
    var has = False
    if i < Int(live_count[unsafe_offset=0]) * SHADOW_SLOTS:
        t = Int(live_idx[unsafe_offset=i // SHADOW_SLOTS]) * SHADOW_SLOTS + i % SHADOW_SLOTS
        has = tasks[unsafe_offset=t].active != Int32(0)
    var pos = warp_append(counter, has)
    if has:
        ids[unsafe_offset=pos] = Int32(t)

# Adds the rays that reached their light (marked 2 by sw_resolve_shadow_gpu) and frees every slot of the live paths.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def sw_add_shadow_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(live_count[unsafe_offset=0]):
        return
    var p = Int(live_idx[unsafe_offset=i])
    comptime for s in range(SHADOW_SLOTS):
        var t = p * SHADOW_SLOTS + s
        if tasks[unsafe_offset=t].active == Int32(2):
            paths[unsafe_offset=p].estimate += tasks[unsafe_offset=t].contrib
        tasks[unsafe_offset=t].active = Int32(0)

# any_hit_bvh2_core with the curve / instance / alpha code compiled out when the scene has none (as traverse_paths_gpu does
# for the closest hit): the full version needs 112 registers, this one far fewer, so more warps stay resident.
@always_inline
def any_hit_bvh2_gpu[curves_on: Bool, inst_on: Bool, alpha_on: Bool](
    sd: SceneView,
    ray: Ray,
    tMax: Float32,
) -> Bool:
    for i in range(Int(sd.sphereCount)):
        if _shadow_is_null_material(sd.materials, Int64(sd.spheres[unsafe_offset=i].materialIndex)):
            continue
        if ray_sphere_hit(sd.spheres[unsafe_offset=i].center, sd.spheres[unsafe_offset=i].radius, ray, Float32(1e-4), tMax) > Float32(0.0):
            return True
    var rdir = Vec3f(Float32(1.0) / ray.direction.x, Float32(1.0) / ray.direction.y, Float32(1.0) / ray.direction.z)
    var org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var nearXIsMin = rdir.x >= Float32(0.0)
    var nearYIsMin = rdir.y >= Float32(0.0)
    var nearZIsMin = rdir.z >= Float32(0.0)
    var inst_leaf = PrimId(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var stack = Array[Int32, 64](uninitialized=True)
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0
    var nodes_f = sd.bvh2Nodes.unsafe_bitcast[Float32]()
    var cur_oc = _node_oc(_node_hi(nodes_f, 0))
    var ray_org = org
    var ray_dir = Vec3f(ray.direction.x, ray.direction.y, ray.direction.z)
    var has_nodes = _bvh_has_nodes(cur_oc)
    while has_nodes:
        if cur_oc[1] > 0:
            var offset = Int(cur_oc[0])
            var count = Int(cur_oc[1])
            for j in range(count):
                var prim = sd.primIds[unsafe_offset=offset + j]
                var mesh_idx: Int
                var base_vidx: Int
                if prim.type == 0:
                    mesh_idx = Int(prim.id1)
                    base_vidx = Int(prim.id2)
                elif prim.type == 1 or prim.type == 2 or prim.type == 3:
                    if prim.id2 == -1:
                        continue
                    mesh_idx = Int(prim.id2 >> 32)
                    base_vidx = Int(prim.id2 & 0xFFFFFFFF) * 3
                elif curves_on and prim.type == 5:
                    if _shadow_is_null_material(sd.materials, prim.materialIndex):
                        continue
                    var curve = sd.curves[unsafe_offset=Int(prim.id1)]
                    if intersect_curve(ray_org, ray_dir, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, tMax)[0]:
                        return True
                    continue
                elif inst_on and prim.type == 8:
                    if not _shadow_is_null_material(sd.materials, prim.materialIndex):
                        inst_leaf = prim      # walked after this loop, so a warp's threads walk the instances together
                    continue
                else:
                    continue
                var v0 = mesh_vi(sd.meshes[unsafe_offset=mesh_idx], base_vidx)
                var v1 = mesh_vi(sd.meshes[unsafe_offset=mesh_idx], base_vidx + 1)
                var v2 = mesh_vi(sd.meshes[unsafe_offset=mesh_idx], base_vidx + 2)
                var p0 = mesh_pos(sd.meshes[unsafe_offset=mesh_idx], v0)
                var p1 = mesh_pos(sd.meshes[unsafe_offset=mesh_idx], v1)
                var p2 = mesh_pos(sd.meshes[unsafe_offset=mesh_idx], v2)
                var hit_res = intersect_triangle(ray_org, ray_dir, p0, p1, p2, tMax)
                if hit_res[0]:
                    if _shadow_is_null_material(sd.materials, prim.materialIndex):
                        continue
                    comptime if alpha_on:
                        var mesh = sd.meshes[unsafe_offset=mesh_idx]
                        if alpha_killed(mesh, v0, v1, v2, hit_res[2], hit_res[3], ray_org, ray_dir, (mesh_idx << 32) | base_vidx):
                            continue
                    return True
            if toVisit == 0:
                break
            toVisit -= 1
            current = Int(stack_ptr[unsafe_offset=toVisit])
            cur_oc = _node_oc(_node_hi(nodes_f, current))
        else:
            var leftIdx = current + 1
            var rightIdx = Int(cur_oc[0])
            var leftLo = _node_lo(nodes_f, leftIdx)
            var leftHi = _node_hi(nodes_f, leftIdx)
            var rightLo = _node_lo(nodes_f, rightIdx)
            var rightHi = _node_hi(nodes_f, rightIdx)
            var leftHit = _node_box_hit(leftLo, leftHi, rdir, org, nearXIsMin, nearYIsMin, nearZIsMin, tMax)
            var rightHit = _node_box_hit(rightLo, rightHi, rdir, org, nearXIsMin, nearYIsMin, nearZIsMin, tMax)
            var leftIsHit = leftHit[0]
            var rightIsHit = rightHit[0]
            if leftIsHit and rightIsHit:
                if leftHit[1] <= rightHit[1]:
                    current = leftIdx
                    cur_oc = _node_oc(leftHi)
                    stack_ptr[unsafe_offset=toVisit] = Int32(rightIdx)
                else:
                    current = rightIdx
                    cur_oc = _node_oc(rightHi)
                    stack_ptr[unsafe_offset=toVisit] = Int32(leftIdx)
                toVisit += 1
            elif leftIsHit:
                current = leftIdx
                cur_oc = _node_oc(leftHi)
            elif rightIsHit:
                current = rightIdx
                cur_oc = _node_oc(rightHi)
            else:
                if toVisit == 0:
                    break
                toVisit -= 1
                current = Int(stack_ptr[unsafe_offset=toVisit])
                cur_oc = _node_oc(_node_hi(nodes_f, current))
    comptime if inst_on:
        if inst_leaf.type == Int8(8):
            return _traverse_instances[True](inst_leaf, sd.meshes, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, ray_org, ray_dir, tMax, True, sd.curves)[0]
    return False

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_TRAVERSE))
def sw_resolve_shadow_gpu[rare_on: Bool, alpha_on: Bool](
    sd: SceneView,
    tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    ids: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(counter[unsafe_offset=0]):
        return
    var t = Int(ids[unsafe_offset=i])
    var task = tasks[unsafe_offset=t]
    var ray = Ray(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
    tasks[unsafe_offset=t].active = Int32(0) if any_hit_bvh2_gpu[rare_on, rare_on, alpha_on](sd, ray, task.tmax) else Int32(2)

# Lists the active deferred shadow rays of ALL slots (task index = path * SHADOW_SLOTS + slot) as dense rays.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def rt_pack_shadow_compact_kernel(
    tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    ids: Pointer[Int32, MutUntrackedOrigin],
    counter: Pointer[Int32, MutUntrackedOrigin],
    count_dp: Int64,
    cap_dp: Int64,
    live_ids: Pointer[Int32, MutUntrackedOrigin],
):
    # One thread per (live path, slot): only paths that were live this round can have queued a task.
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var t = 0
    var live = False
    if i < Int(count_dp):
        t = Int(live_ids[unsafe_offset=i // SHADOW_SLOTS]) * SHADOW_SLOTS + i % SHADOW_SLOTS
        live = tasks[unsafe_offset=t].active == Int32(1)
    var pos = warp_append(counter, live)
    if not live or pos >= Int(cap_dp):           # beyond the capacity: stays active for the next round of the caller's loop
        return
    var task = tasks[unsafe_offset=t]
    var idx = pos * 8
    rays[unsafe_offset=idx + 0] = task.origin.x
    rays[unsafe_offset=idx + 1] = task.origin.y
    rays[unsafe_offset=idx + 2] = task.origin.z
    rays[unsafe_offset=idx + 3] = Float32(0)
    rays[unsafe_offset=idx + 4] = task.direction.x
    rays[unsafe_offset=idx + 5] = task.direction.y
    rays[unsafe_offset=idx + 6] = task.direction.z
    rays[unsafe_offset=idx + 7] = task.tmax
    ids[unsafe_offset=pos] = Int32(t)

# Frees every listed shadow ray's task for the next bounce (so the tasks never need a full-grid reset between bounces) and
# lists, per slot, the indices of the rays no triangle or sphere blocks: the curve walk then runs on full warps. Slot 0's list grows up
# from `lists`, slot 1's down from lists + cap; together they hold at most the `count` rays. Bit 31: too many rejected
# alpha hits for the RT-core passes, the software BVH decides.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def shadow_survivors_kernel(
    tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    raw: Pointer[UInt32, MutUntrackedOrigin],
    ids: Pointer[Int32, MutUntrackedOrigin],
    sd: SceneView,
    n_spheres_dp: Int64,
    count_dp: Int64,
    converted_dp: Int64,
    lists: Pointer[Int32, MutUntrackedOrigin],
    counts: Pointer[Int32, MutUntrackedOrigin],
    cap_dp: Int64,
):
    comptime assert SHADOW_SLOTS <= 2, "a list per slot, from either end"
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var t = 0
    var keep = False
    var soft = False
    if i < Int(count_dp):
        t = Int(ids[unsafe_offset=i])
        tasks[unsafe_offset=t].active = Int32(0)
        var flag = raw[unsafe_offset=i * 8 + 6]
        soft = converted_dp != Int64(0) and flag == UInt32(3)
        keep = soft or not (flag != UInt32(0) if converted_dp != Int64(0) else raw[unsafe_offset=i * 8 + 3] != UInt32(0xffffffff))
        if keep and not soft:                  # analytic spheres are not in the acceleration structure
            var task = tasks[unsafe_offset=t]
            var ray = Ray(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
            for k in range(Int(n_spheres_dp)):
                if ray_sphere_hit(sd.spheres[unsafe_offset=k].center, sd.spheres[unsafe_offset=k].radius, ray, Float32(1e-4), task.tmax) > Float32(0.0):
                    keep = False
                    break
    var slot = t % SHADOW_SLOTS
    var p0 = warp_append(counts, keep and slot == 0)
    barrier()                                  # warp_append's shared counts, read above, are used again
    var p1 = warp_append(counts.unsafe_offset(1), keep and slot == 1)
    if keep:
        var e = Int32(i) | (Int32(1) << 31 if soft else Int32(0))
        lists[unsafe_offset=p0 if slot == 0 else Int(cap_dp) - 1 - p1] = e


# Adds the contribution of every listed shadow ray of one slot that no curve blocks either. Slot by slot, so a path's two
# rays never add at once, and in slot order, which keeps the sum order.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def shadow_curves_kernel(
    paths: Pointer[PathState, MutUntrackedOrigin],
    tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    sd: SceneView,
    n_spheres_dp: Int64,
    ids: Pointer[Int32, MutUntrackedOrigin],
    lists: Pointer[Int32, MutUntrackedOrigin],
    counts: Pointer[Int32, MutUntrackedOrigin],
    slot_dp: Int64,
    cap_dp: Int64,
    blocked: Pointer[UInt8, MutUntrackedOrigin],               # the OptiX curves' verdicts per ray, else dangling
):
    var j = Int(block_idx.x * block_dim.x + thread_idx.x)
    if j >= Int(counts[unsafe_offset=Int(slot_dp)]):
        return
    var e = lists[unsafe_offset=j if slot_dp == Int64(0) else Int(cap_dp) - 1 - j]
    var i = Int(e & Int32(0x7FFFFFFF))
    var t = Int(ids[unsafe_offset=i])
    var task = tasks[unsafe_offset=t]
    var ray = Ray(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
    if e < Int32(0):
        if any_hit_bvh2_core[True](sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, task.tmax, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, Int(n_spheres_dp), materials=sd.materials):
            return
        if _curves_apart(sd) and curve_bvh_occluded(sd.curveBvh, sd.curves, sd.materials, ray, task.tmax):
            return
    elif _is_real_ptr(blocked):
        if blocked[unsafe_offset=i] != UInt8(0):
            return
    elif sd.curveBvh.n_clusters > Int32(0):
        if curve_bvh_occluded(sd.curveBvh, sd.curves, sd.materials, ray, task.tmax):
            return
    elif Int(sd.curveCount) > 0 and any_hit_bvh2_core[True](sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, task.tmax, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, 0, materials=sd.materials, root=max(Int(sd.curveRoot), 0)):
        return                                                     # curves are not in the acceleration structure either
    paths[unsafe_offset=t // SHADOW_SLOTS].estimate += task.contrib

comptime SHADOW_COUNTS = 16        # in the counter buffer, after the shading kinds' counts: the survivors per slot
comptime SSS_RT_COUNTS = 20        # in the counter buffer: the walkers of the RT steps, alternately
comptime SSS_RT_STEPS = 16         # walk steps a round with RT-core segments, while enough walkers remain
comptime SSS_RT_MIN = 4096         # fewer walkers finish in software: the tail of long walks


def rtcore_sss_walk_gpu(
    ctx: DeviceContext,
    paths: Pointer[PathState, MutUntrackedOrigin],
    inter: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    rays_buf: DeviceBuffer[DType.float32],
    results_buf: DeviceBuffer[DType.float32],
    ids_buf: DeviceBuffer[DType.uint8],
    counter_buf: DeviceBuffer[DType.uint8],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin],
    walkers: Int,                  # counter slot of the walker list at 3 n_total (compact_walking_gpu's)
    n_total: Int,
) raises:
    """The walks of the listed walkers, their segments traced on the RT cores a step at a time, so the walkers of a step
    are full warps and no lane waits on another's long walk; the few walks still going after SSS_RT_STEPS steps, or
    once fewer than SSS_RT_MIN remain, finish in software (sss_walk_gpu)."""
    comptime block_size = 256
    var rt_hw = rtcore_active()
    var cuda_stream = CUDA(ctx.stream())
    var rays = rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var results = results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var ids_base = ids_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var counts = counter_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var lists = [ids_base.unsafe_offset(3 * n_total), ids_base.unsafe_offset(2 * n_total)]   # the shadow ids' room, free now
    var slots = [walkers, SSS_RT_COUNTS]
    var m = n_total
    for step in range(SSS_RT_STEPS):
        var a = step % 2
        var out_slot = SSS_RT_COUNTS + step % 2
        ctx.enqueue_memset(DeviceBuffer[DType.int32](ctx, counts.unsafe_offset(out_slot), 1, owning=False), Int32(0))
        ctx.enqueue_function[sss_walk_rt_step_gpu](paths, inter, sd, lists[a], counts.unsafe_offset(slots[a]), lists[1 - a],
                                                   counts.unsafe_offset(out_slot), rays, grid_dim=ceildiv(m, block_size), block_dim=block_size)
        m = read_counter(ctx, counter_buf, out_slot)
        slots[1 - a] = out_slot
        if m == 0:
            return
        _ = rtcore_trace_interop(rt_hw, UInt64(Int(rays)), UInt64(Int(results)), Int32(m), cuda_stream)
        ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
            results, inter,
            mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(n_meshes), Int64(m), instance_base_mesh, lists[1 - a], rt_no_paths(), sd.meshes,
            grid_dim=ceildiv(m, block_size), block_dim=block_size)
        if m < SSS_RT_MIN or step == SSS_RT_STEPS - 1:
            ctx.enqueue_function[sss_walk_gpu](paths, inter, sd, lists[1 - a], counts.unsafe_offset(slots[1 - a]),
                                               grid_dim=ceildiv(m, block_size), block_dim=block_size)
            return


def rtcore_shadow_rays_compact_gpu(
    ctx: DeviceContext,
    path_buf: DeviceBuffer[DType.uint8],
    shadow_buf: DeviceBuffer[DType.uint8],
    interop_rays_buf: DeviceBuffer[DType.float32],
    interop_results_buf: DeviceBuffer[DType.float32],
    ids_buf: DeviceBuffer[DType.uint8],
    counter_buf: DeviceBuffer[DType.uint8],
    n_total: Int,
    sd: SceneView,
    n_spheres: Int,
    scratch_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin],
    round: Int,
    n_live: Int,
    bench: Pointer[Int, MutUntrackedOrigin],
) raises:
    comptime block_size = 256
    var n_tasks = n_live * SHADOW_SLOTS
    var cuda_stream = CUDA(ctx.stream())
    var rt_hw = rtcore_active()
    var alpha = Int(rtcore_alpha_enabled()) != 0
    var tasks = shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var rays = interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var ids_base = ids_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var live_ids = ids_base.unsafe_offset((round % 2) * n_total)
    var ids = ids_base.unsafe_offset(2 * n_total)
    while True:
        ctx.enqueue_memset(counter_buf, UInt8(0))
        ctx.enqueue_function[rt_pack_shadow_compact_kernel](
            tasks, rays, ids, counter_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), Int64(n_tasks), Int64(n_total), live_ids,
            grid_dim=ceildiv(n_tasks, block_size), block_dim=block_size,
        )
        var found = read_counter(ctx, counter_buf)
        var m = min(found, n_total)
        if m == 0:
            return
        bench[unsafe_offset=1] += m
        if bench[unsafe_offset=7] != 0:
            _ray_dump(ctx, interop_rays_buf, interop_results_buf, m, "s" + String(round))
        var t0 = perf_counter_ns()
        if alpha:
            # Interop layout, so the alpha passes know which triangle each ray hit.
            _ = rtcore_trace_interop(rt_hw, UInt64(Int(interop_rays_buf.unsafe_ptr())), UInt64(Int(interop_results_buf.unsafe_ptr())), Int32(m), cuda_stream)
            rtcore_alpha_passes(ctx, interop_rays_buf, interop_results_buf, scratch_buf, sd.meshes, n_meshes, instance_base_mesh, m)
        else:
            _ = rtcore_trace(rt_hw, UInt64(Int(interop_rays_buf.unsafe_ptr())), UInt64(Int(interop_results_buf.unsafe_ptr())), Int32(m), cuda_stream)
        var counts = counter_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_offset(SHADOW_COUNTS)
        var lists = ids_base.unsafe_offset(3 * n_total)          # the shading kinds' list, done with by now
        ctx.enqueue_function[shadow_survivors_kernel](
            tasks, interop_results_buf.unsafe_ptr().unsafe_bitcast[UInt32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            ids, sd, Int64(n_spheres), Int64(m), Int64(1 if alpha else 0), lists, counts, Int64(n_total),
            grid_dim=ceildiv(m, block_size), block_dim=block_size,
        )
        var ox = oxc_active()
        var blocked = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
        if Int(ox) != 0 and Int(sd.curveCount) > 0:
            var b = oxc_scratch(ox, m)
            if b != UInt64(0):
                blocked = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(b))
                for s in range(SHADOW_SLOTS):
                    _ = oxc_trace_shadow(ox, UInt64(Int(rays)), UInt64(Int(lists)), UInt64(Int(counts)), n_total, s, b, m, cuda_stream)
        for s in range(SHADOW_SLOTS):
            ctx.enqueue_function[shadow_curves_kernel](
                path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                tasks, sd, Int64(n_spheres), ids, lists, counts, Int64(s), Int64(n_total), blocked,
                grid_dim=ceildiv(m, CURVE_BLOCK), block_dim=CURVE_BLOCK,
            )
        if bench[unsafe_offset=3] == 2:
            ctx.synchronize()
            bench[unsafe_offset=2] += Int(perf_counter_ns() - t0)
        if found <= n_total:
            return


# The Vulkan ray-query shader treats every triangle as opaque. A ray whose nearest hit is on a mesh with an alpha
# cutout (or a constant alpha below 1, as on an invisible emitter) is traced again in software, which honours it.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vk_alpha_retrace_gpu(
    sd: SceneView,
    paths: Pointer[PathState, MutUntrackedOrigin],
    inter: Pointer[Intersection, MutUntrackedOrigin],
    count_dp: Int64,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(count_dp) or paths[unsafe_offset=tid].active == Int8(0) or inter[unsafe_offset=tid].hit == Int8(0):
        return
    var prim = inter[unsafe_offset=tid].primId
    var mi = -1
    if prim.type == Int8(0):
        mi = Int(prim.id1)
    elif prim.type == Int8(3):
        mi = Int(prim.id2 >> 32)
    if mi < 0:
        return
    var mesh = sd.meshes[unsafe_offset=mi]
    if mesh.alpha_w <= Int32(0) and mesh.alpha_const >= Float32(1.0):
        return
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, paths[unsafe_offset=tid].ray, Float32(1.0e38),
                       inter.unsafe_offset(tid), sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    if _curves_apart(sd):
        var t = inter[unsafe_offset=tid].tHit if inter[unsafe_offset=tid].hit != Int8(0) else Float32(1.0e38)
        var h = curve_bvh_closest(sd.curveBvh, sd.curves, paths[unsafe_offset=tid].ray, t)
        if h[0] >= 0:
            inter[unsafe_offset=tid] = Intersection(curve_bvh_prim(sd.curveBvh, h[0]), h[1], h[2], h[3], Int8(1), Int8(0), Int8(0), Int8(0))


def vulkaninterop_rt_traverse_paths_gpu(
    ctx: DeviceContext,
    path_buf: DeviceBuffer[DType.uint8],
    inter_buf: DeviceBuffer[DType.uint8],
    sd: SceneView,
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    interop_results_buf: DeviceBuffer[DType.float32],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    spheres: Pointer[Sphere, MutUntrackedOrigin] = Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
    # --rt-hardware with alpha cutouts (see rtcore_alpha_passes).
    rt_scratch_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    alpha: Bool = False,   # the scene has alpha surfaces (vk_alpha_retrace_gpu)
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)
    ctx.enqueue_function[vulkaninterop_pack_rays_kernel](
        path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_total),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    # --rt-hardware: trace on the RT cores from a CUDA kernel (docs/rtcore/NOTES.md) instead of dispatching the Vulkan
    # ray-query shader. Same rays buffer, same Result layout.
    var rt_hw = rtcore_active()
    if Int(rt_hw) != 0 and rt_scratch_buf:
        rtcore_trace_unpack_gpu(ctx, inter_buf, sd, interop_rays_buf, interop_results_buf, rt_scratch_buf.value(),
                                mesh_material_idx_buf, mesh_al_idx_buf, n_meshes, n_total, instance_base_mesh_buf)
    else:
        _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)
        var instance_base_mesh_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        if instance_base_mesh_buf:
            instance_base_mesh_ptr = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
            interop_results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(n_meshes),
            Int64(n_total),
            instance_base_mesh_ptr,
            rt_no_ids(), rt_no_paths(), rt_no_meshes(),
            grid_dim=grid, block_dim=block_size,
        )
        if alpha:
            ctx.enqueue_function[vk_alpha_retrace_gpu](
                sd,
                path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                Int64(n_total),
                grid_dim=grid, block_dim=block_size,
            )

    # Curves need no further handling here at all -- intersect_batch.comp
    # already resolved them (real narrow-phase test + generate on a valid
    # hit) as part of the SAME dispatch, and vulkaninterop_unpack_results_
    # kernel above already decoded a curve hit (hitFlag==2) into inter_buf.

    if n_spheres > 0:
        ctx.enqueue_function[vulkaninterop_test_spheres_gpu](
            path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            spheres,
            Int64(n_spheres),
            Int64(n_total),
            grid_dim=grid, block_dim=block_size,
        )


# ── Curve-divergence-mitigation kernels (companions to traverse_paths_gpu) ────
# See traverse_bvh2_core_defer_curves for the rationale. This is a 3-kernel
# pipeline run once per bounce, only when the scene has curves:
#   1. traverse_paths_gpu records up to CURVE_DEFER_K candidate curve prims per
#      ray instead of testing them inline (above).
#   2. compact_curve_paths_gpu streams the (typically sparse) subset of rays
#      that recorded >=1 candidate into a dense array via a single atomic append
#      per curve-touching ray — cheap even though divergent, since there's no
#      expensive math in this kernel, just a compare-and-maybe-atomic.
#   3. resolve_curve_candidates_gpu processes that dense array: each thread owns
#      exactly one ray (compaction guarantees no ray appears twice), so there is
#      no race updating `results`, and warps are densely packed with real work
#      instead of ~92% idle lanes.

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def reset_curve_counter_gpu(counter: Pointer[Int32, MutUntrackedOrigin]):
    if block_idx.x == 0 and thread_idx.x == 0:
        counter[unsafe_offset=0] = Int32(0)

# The CUDA-native path (traverse_bvh2_core_defer_curves) always writes
# curve candidates at tid*CURVE_DEFER_K -- this reproduces that formula once
# at buffer-creation time so curve_cand_offset_buf is valid immediately.
# Vulkan RT rendering never touches curve_cand_prim_buf/curve_cand_count_buf/
# curve_cand_offset_buf at all anymore -- intersect_batch.comp resolves
# curve hits itself and writes them straight into the ordinary results
# buffer (see vulkaninterop_unpack_results_kernel's hitFlag==2 branch), so
# compact_curve_paths_gpu/resolve_curve_candidates_gpu never run for a
# Vulkan-RT render (see _gpu_bounce_kernels).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def compact_curve_paths_gpu(
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    n_dp: Int64,
    compact_pathIds: Pointer[Int32, MutUntrackedOrigin],
    compact_counter: Pointer[Int32, MutUntrackedOrigin],
):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if curve_cand_count[unsafe_offset=tid] > Int32(0):
        var pos = Atomic.fetch_add(compact_counter, Int32(1))
        compact_pathIds[unsafe_offset=Int(pos)] = Int32(tid)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def resolve_curve_candidates_gpu(
    compact_pathIds: Pointer[Int32, MutUntrackedOrigin],
    compact_counter: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_prim: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_offset: Pointer[Int32, MutUntrackedOrigin],
    sd: SceneView,
    paths: Pointer[PathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    n_dp: Int64,
):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if tid >= Int(compact_counter[unsafe_offset=0]):
        return
    var pathId = Int(compact_pathIds[unsafe_offset=tid])
    var n_cand = Int(curve_cand_count[unsafe_offset=pathId])
    if n_cand == 0:
        return
    var base = Int(curve_cand_offset[unsafe_offset=pathId])
    var ray = paths[unsafe_offset=pathId].ray
    var ray_org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var ray_dir = Vec3f(ray.direction.x, ray.direction.y, ray.direction.z)
    var res = results[unsafe_offset=pathId]
    var best_t = res.tHit
    var best_u = res.u
    var best_v = res.v
    var best_prim = res.primId
    var best_hit = res.hit
    var changed = False
    for i in range(n_cand):
        var primIdx = Int(curve_cand_prim[unsafe_offset=base + i])
        var prim = sd.primIds[unsafe_offset=primIdx]
        var curve = sd.curves[unsafe_offset=Int(prim.id1)]
        var curve_hit = intersect_curve(ray_org, ray_dir, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, best_t)
        if curve_hit[0]:
            best_t = curve_hit[1]
            best_u = curve_hit[2]
            best_v = curve_hit[3]
            best_prim = prim
            best_hit = Int8(1)
            changed = True
    if changed:
        results[unsafe_offset=pathId] = Intersection(best_prim, best_t, best_u, best_v, best_hit, 0, 0, 0)


# GPU kernel: generate primary PathState for every pixel in one pass.
# Each thread handles one pixel.  All sampling is pure math — no host calls.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def gen_primary_rays_gpu(
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    paths: Pointer[PathState, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count_dp: Int64,
    lens_radius: Float32, focal_distance: Float32,
):
    var fw = Int(fw_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var px = Int32(tid % fw)
    var py = Int32(tid // fw)
    var rng_seed = UInt64(rng_seed_hi) << UInt64(32) | UInt64(rng_seed_lo)
    var (ray, pcg_state, pcg_inc, sobol_idx, wavelengths) = gen_primary_ray_state(
        px, py, si, Int(log2spp), Int(n_base4),
        seed_dim0, seed_dim1, rng_seed, sobol_matrices, r2c, c2w,
        filter_norm_x, filter_sigma, filter_support_x,
        filter_norm_y, filter_support_y,
        filter_type,
        lens=ThinLens(lens_radius, focal_distance),
    )
    paths[unsafe_offset=tid] = PathState(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Float32(1.0),   # previous_dielectric_ior (vacuum)
        Float32(1.0),   # eta_scale
        ThinLens(lens_radius, focal_distance).first_dim(), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
        INV_FOUR_PI,    # lastEnvNeePdf (gated by lastBsdfPdf > 0; set at each scatter)
        Float32(0.0),   # cone_len: total path length, accumulated per bounce
        Int8(0),   # ptex_missed
        MIN_RAY_PDF_NONE,   # min_ray_pdf
        SssWalk.none(),
    )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def pack_center_rays_gpu(
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
):
    var fw = Int(fw_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * Int(fh_dp):
        return
    var filmX = Float32(tid % fw) + Float32(0.5)
    var filmY = Float32(tid // fw) + Float32(0.5)
    var cx = r2c[unsafe_offset=0]*filmX + r2c[unsafe_offset=4]*filmY + r2c[unsafe_offset=12]
    var cy = r2c[unsafe_offset=1]*filmX + r2c[unsafe_offset=5]*filmY + r2c[unsafe_offset=13]
    var cz = r2c[unsafe_offset=2]*filmX + r2c[unsafe_offset=6]*filmY + r2c[unsafe_offset=14]
    var cw = r2c[unsafe_offset=3]*filmX + r2c[unsafe_offset=7]*filmY + r2c[unsafe_offset=15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = sqrt(cx*cx + cy*cy + cz*cz)
    if cl > Float32(0): cx /= cl; cy /= cl; cz /= cl
    var dir = Vec3f(
        c2w[unsafe_offset=0]*cx + c2w[unsafe_offset=4]*cy + c2w[unsafe_offset=8]*cz,
        c2w[unsafe_offset=1]*cx + c2w[unsafe_offset=5]*cy + c2w[unsafe_offset=9]*cz,
        c2w[unsafe_offset=2]*cx + c2w[unsafe_offset=6]*cy + c2w[unsafe_offset=10]*cz,
    )
    var dl = dir.length()
    if dl > Float32(0): dir = dir / dl
    var idx = tid * 8
    rays[unsafe_offset=idx + 0] = c2w[unsafe_offset=12]
    rays[unsafe_offset=idx + 1] = c2w[unsafe_offset=13]
    rays[unsafe_offset=idx + 2] = c2w[unsafe_offset=14]
    rays[unsafe_offset=idx + 3] = Float32(1e-4)
    rays[unsafe_offset=idx + 4] = dir.x
    rays[unsafe_offset=idx + 5] = dir.y
    rays[unsafe_offset=idx + 6] = dir.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8)


# GPU kernel: shoot one unjittered center ray per pixel and write normals + depth.
# Used to guide the à-trous denoiser with geometric edge information.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def gen_aux_buffers_gpu(
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    curves: Pointer[Curve, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin],
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    isects_tmp: Pointer[Intersection, MutUntrackedOrigin],
    normals_out: Pointer[Float32, MutUntrackedOrigin],
    depth_out: Pointer[Float32, MutUntrackedOrigin],
    curve_mask_out: Pointer[Float32, MutUntrackedOrigin],
    world_pos_out: Pointer[Float32, MutUntrackedOrigin],
    material_id_out: Pointer[Int32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    hits_traced: Int32 = Int32(0),   # isects_tmp already holds this ray's hit (traced on the RT cores)
):
    var n_spheres = Int(n_spheres_dp)
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw
    var py = tid // fw
    var filmX = Float32(px) + Float32(0.5)
    var filmY = Float32(py) + Float32(0.5)

    # Raster → camera
    var cx = r2c[unsafe_offset=0]*filmX + r2c[unsafe_offset=4]*filmY + r2c[unsafe_offset=12]
    var cy = r2c[unsafe_offset=1]*filmX + r2c[unsafe_offset=5]*filmY + r2c[unsafe_offset=13]
    var cz = r2c[unsafe_offset=2]*filmX + r2c[unsafe_offset=6]*filmY + r2c[unsafe_offset=14]
    var cw = r2c[unsafe_offset=3]*filmX + r2c[unsafe_offset=7]*filmY + r2c[unsafe_offset=15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = sqrt(cx*cx + cy*cy + cz*cz)
    if cl > Float32(0): cx /= cl; cy /= cl; cz /= cl

    # Camera → world
    var dir = Vec3f(
        c2w[unsafe_offset=0]*cx + c2w[unsafe_offset=4]*cy + c2w[unsafe_offset=8]*cz,
        c2w[unsafe_offset=1]*cx + c2w[unsafe_offset=5]*cy + c2w[unsafe_offset=9]*cz,
        c2w[unsafe_offset=2]*cx + c2w[unsafe_offset=6]*cy + c2w[unsafe_offset=10]*cz,
    )
    var dl = dir.length()
    if dl > Float32(0): dir = dir / dl
    var org = Point3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])

    var ray = Ray(org, dir)
    if hits_traced == Int32(0):
        var dummy_id = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
        isects_tmp[unsafe_offset=tid] = Intersection(dummy_id, Float32(1e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
        traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, Float32(1e38), isects_tmp.unsafe_offset(tid), blasNodesArr, blasPrimIdsArr, instances)
    test_spheres(spheres, n_spheres, ray, isects_tmp.unsafe_offset(tid))

    var normal = Vec3f(Float32(0), Float32(0), Float32(1))
    var d = Float32(1e38)

    if isects_tmp[unsafe_offset=tid].hit != Int8(0):
        d = isects_tmp[unsafe_offset=tid].tHit
        var typ = Int(isects_tmp[unsafe_offset=tid].primId.type)
        if typ == 4:
            var si = Int(isects_tmp[unsafe_offset=tid].primId.id1)
            normal = sphere_outward_normal(org + dir*d, spheres[unsafe_offset=si].center)
        elif typ == 5:
            # Approximate outward normal for the denoiser G-buffer: h alone
            # (stored in isects_tmp.u) doesn't uniquely fix the azimuthal sign,
            # so this picks one consistent side — fine for denoising, not used
            # for shading (shade_hair derives its own frame independently).
            var curve = curves[unsafe_offset=Int(isects_tmp[unsafe_offset=tid].primId.id1)]
            var piece = min(Int(curve.n_pieces) - 1, Int(isects_tmp[unsafe_offset=tid].v * Float32(curve.n_pieces)))
            var (cq0, cq1, _, _) = curve_piece_endpoints(curve, piece)
            var caxis = cq1 - cq0
            var calen = sqrt(dot(caxis, caxis))
            if calen > Float32(1e-8):
                var ctangent = caxis * (Float32(1.0) / calen)
                var cn = curve_hit_normal(ctangent, -dir, isects_tmp[unsafe_offset=tid].u)
                normal = vec3f(cn)
        elif typ == 0 or typ == 1 or typ == 2 or typ == 3:
            var mesh_idx: Int
            var base_vidx: Int
            if typ == 0:
                mesh_idx  = Int(isects_tmp[unsafe_offset=tid].primId.id1)
                base_vidx = Int(isects_tmp[unsafe_offset=tid].primId.id2)
            else:
                mesh_idx  = Int(isects_tmp[unsafe_offset=tid].primId.id2 >> 32)
                base_vidx = Int(isects_tmp[unsafe_offset=tid].primId.id2 & 0xFFFFFFFF) * 3
            var mesh = meshes[unsafe_offset=mesh_idx]
            var vi0 = Int(mesh_vi(mesh, base_vidx))
            var vi1 = Int(mesh_vi(mesh, base_vidx + 1))
            var vi2 = Int(mesh_vi(mesh, base_vidx + 2))
            var p0 = mesh_point(mesh, vi0)
            var p1 = mesh_point(mesh, vi1)
            var p2 = mesh_point(mesh, vi2)
            var e1 = p1 - p0; var e2 = p2 - p0
            normal = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
            var inst_idx = isects_tmp[unsafe_offset=tid].primId.instanceIdx
            if inst_idx >= Int32(0):
                var n_world = transform_normal(Mat4(instances[unsafe_offset=Int(inst_idx)].world_to_obj()), normal.to_simd())
                normal = vec3f(n_world)
            var nl = normal.length()
            if nl > Float32(0): normal = normal / nl
        # else: unrecognized primitive type (shouldn't happen — every hit
        # traverse_bvh2_core can produce is type 0-5) — leave normal at the
        # miss-ray default (0,0,1) rather than misreading id1/id2 as mesh data.
        if normal.dot(-dir) < Float32(0):
            normal = -normal

    normals_out[unsafe_offset=tid*3+0] = normal.x
    normals_out[unsafe_offset=tid*3+1] = normal.y
    normals_out[unsafe_offset=tid*3+2] = normal.z
    depth_out[unsafe_offset=tid] = d
    curve_mask_out[unsafe_offset=tid] = Float32(1.0) if (isects_tmp[unsafe_offset=tid].hit != Int8(0) and Int(isects_tmp[unsafe_offset=tid].primId.type) == 5) else Float32(0.0)
    store_vec3(world_pos_out, tid, (org + dir*d).to_simd())
    material_id_out[unsafe_offset=tid] = Int32(isects_tmp[unsafe_offset=tid].primId.materialIndex) if isects_tmp[unsafe_offset=tid].hit != Int8(0) else Int32(-1)


def gpu_gen_aux_buffers[Oc: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    c2w: Pointer[Float32, Oc],
    n: Int64,
    # --rt-hardware: the center rays are traced on the RT cores, the CUDA BVH does not hold the scene.
    rt_rays: Optional[DeviceBuffer[DType.float32]] = None,
    rt_results: Optional[DeviceBuffer[DType.float32]] = None,
    rt_mesh_material_idx: Optional[DeviceBuffer[DType.uint8]] = None,
    rt_mesh_al_idx: Optional[DeviceBuffer[DType.uint8]] = None,
    rt_n_meshes: Int = 0,
    rt_instance_base_mesh: Optional[DeviceBuffer[DType.uint8]] = None,
):
    """Generate unjittered normals and depth buffers for the denoiser."""
    var n_pix = Int(n)
    if n_pix == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            handle[].ctx.enqueue_copy(handle[].c2w_buf, c2w.unsafe_bitcast[UInt8]())
            comptime block_size = 256
            var grid_n = ceildiv(n_pix, block_size)
            var rt_hits = Int32(0)
            if rt_rays:
                handle[].ctx.enqueue_function[pack_center_rays_gpu](
                    handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                    handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                    rt_rays.value().unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                    Int64(handle[].film.width), Int64(handle[].film.height),
                    grid_dim=grid_n, block_dim=block_size,
                )
                rtcore_trace_unpack_gpu(
                    handle[].ctx, handle[].inter_buf, handle[].scene_descriptor(), rt_rays.value(), rt_results.value(),
                    handle[].rt_scratch_buf, rt_mesh_material_idx.value(), rt_mesh_al_idx.value(), rt_n_meshes, n_pix,
                    rt_instance_base_mesh)
                rt_hits = Int32(1)
            handle[].ctx.enqueue_function[gen_aux_buffers_gpu](
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].bvh.nodes_ptr(),
                handle[].bvh.prim_ids_ptr(),
                handle[].meshes.meshes_ptr(),
                handle[].curves.curves_ptr(),
                handle[].blas.nodes_arr(),
                handle[].blas.primids_arr(),
                handle[].scene_descriptor().instances,
                handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere](),
                Int64(handle[].n_spheres),
                handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                handle[].atrous_normals_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_curve_mask_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].gbuf_material_id_buf.unsafe_ptr().unsafe_bitcast[Int32](),
                Int64(handle[].film.width), Int64(handle[].film.height),
                rt_hits,
                grid_dim=grid_n, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU gen aux buffers failed: " + String(e))


# One bounce's worth of kernel dispatch (intersect -> medium -> per-material
# shade), shared verbatim between gpu_render_sample and gpu_render_wavefront
# below -- this used to be duplicated ~380 lines in each. Safe to factor out
# because it is pure host-side enqueue_function orchestration, not GPU device
# code, so it doesn't touch the class of PTX-codegen bugs that blocks sharing
# actual kernel bodies elsewhere in this codebase (see bdpt_*.mojo's MNEE/
# _connect duplication comments for that unrelated, still-real constraint).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def deactivate_paths_past_maxdepth_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    n_dp: Int64, max_depth: Int32,
):
    """A NULL INTERFACE crossing (entering/leaving a medium) does not
    increment `path_ptr[].bounce` -- correctly, since pbrt does not count it
    as a bounce either -- but nothing else previously stopped a path once its
    OWN bounce count reached the scene's real max_depth: the host loop's
    fixed round count (gpu_render_sample/gpu_render_wavefront's own
    `for _ in range(Int(maxDepth))`) was the ENTIRE termination mechanism,
    and every kind of round (real scatter OR free interface pass-through)
    consumed one of those rounds equally. For any volumetric scene, that
    silently granted one fewer REAL bounce than requested for every interface
    the path had to cross -- see rendering.mojo's CPU-side twin of this
    kernel for the measurement. Dispatched as the FIRST kernel of every
    bounce round, so every later kernel's own `active != 0` check already
    skips whatever this one just deactivated."""
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    # Marks, does not kill -- see rendering.mojo's twin of this for why the
    # segment leaving the last allowed vertex must still be traced.
    if paths[unsafe_offset=tid].active != Int8(0) and paths[unsafe_offset=tid].bounce >= max_depth:
        paths[unsafe_offset=tid].at_cap = Int8(1)
