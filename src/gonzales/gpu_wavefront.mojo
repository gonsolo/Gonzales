from .bvh import BVH2Node, any_hit_bvh2_core, test_spheres, traverse_bvh2_core, traverse_bvh2_core_defer_curves, SceneDescriptor2_C
from .curves import CURVE_DEFER_K, Curve_C, _curve_perp_axis, curve_piece_endpoints, intersect_curve
from .geometry import INV_FOUR_PI, Point3f, RGB, Vec3f, _is_real_ptr, cross, dot, store_vec3, vec3f
from .materials import Material_C
from .primitives import Instance_C, Intersection_C, PrimId_C, Ray_C, Sphere_C, TriangleMesh_C, sphere_outward_normal
from .render_state import PathState_C, ShadowTask_C
from .restir_di import DIReservoir, di_reservoir_init
from .restir_vol import VolReservoir, vol_reservoir_init
from .sampling import gen_primary_ray_state
from .spectrum import SpectralSample, spectral_sample_to_rgb
from .transform import transform_normal_by_instance
from .vulkaninterop import VulkanInteropRtSceneHandle, vulkaninterop_rt_trace
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host._nvidia_cuda import CUDA
from std.atomic import Atomic
from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from .gpu_scene import GpuSceneHandle


def reset_shadow_tasks_gpu(
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shadow_tasks[unsafe_offset=tid].active = Int32(0)

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
    sd: SceneDescriptor2_C,
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
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
    var shadow_ray = Ray_C(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
    if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray, task.tmax, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, n_spheres, materials=sd.materials):
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
    multiplies by iso/100 after the fact and pbrt clamps post-sensor.
    lim <= 0 disables it."""
    if lim <= Float32(0):
        return (r, g, b)
    var mx = max(r, max(g, b))
    if mx <= lim:
        return (r, g, b)
    var k = lim / mx
    return (r * k, g * k, b * k)


def accumulate_film_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
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
    albedo_film[unsafe_offset=tid*3+0] += paths[unsafe_offset=tid].albedo.r
    albedo_film[unsafe_offset=tid*3+1] += paths[unsafe_offset=tid].albedo.g
    albedo_film[unsafe_offset=tid*3+2] += paths[unsafe_offset=tid].albedo.b


def clear_film_gpu(film: Pointer[Float32, MutUntrackedOrigin], n_pixels_dp: Int64):
    var n_pixels = Int(n_pixels_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    film[unsafe_offset=tid*3+0] = Float32(0)
    film[unsafe_offset=tid*3+1] = Float32(0)
    film[unsafe_offset=tid*3+2] = Float32(0)


# Wavefront accumulation: thread px sums actual_batch samples from path_buf layout
# path_buf[si * n_pixels + px] and adds to film[px].  No atomics needed (one thread per pixel).
def accumulate_film_wavefront_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
    n_pixels_dp: Int64, actual_batch_dp: Int64,
    sample_clamp: Float32 = Float32(0.0),
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
):
    var n_pixels = Int(n_pixels_dp)
    var actual_batch = Int(actual_batch_dp)
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    if px >= n_pixels:
        return
    var r = Float32(0); var g = Float32(0); var b = Float32(0)
    var ar = Float32(0); var ag = Float32(0); var ab = Float32(0)
    for si in range(actual_batch):
        var p = paths[unsafe_offset=si * n_pixels + px]
        var _pe = spectral_sample_to_rgb(spectral_coeffs, Int(spectral_res_dp),
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
            p.estimate, p.wavelengths)
        var (_qr, _qg, _qb) = _clamp_sample_rgb(_pe[0], _pe[1], _pe[2], sample_clamp)
        r += _qr; g += _qg; b += _qb
        ar += p.albedo.r;  ag += p.albedo.g;  ab += p.albedo.b
    film[unsafe_offset=px*3+0] += r; film[unsafe_offset=px*3+1] += g; film[unsafe_offset=px*3+2] += b
    albedo_film[unsafe_offset=px*3+0] += ar; albedo_film[unsafe_offset=px*3+1] += ag; albedo_film[unsafe_offset=px*3+2] += ab


# Wavefront primary-ray generation: thread ti → pixel (ti % n_pixels), sample (si_start + ti // n_pixels).
# Layout: path_buf[si_local * n_pixels + px_flat] — adjacent threads touch adjacent pixels of same sample.
def gen_primary_rays_wavefront_gpu(
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    si_start: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count_dp: Int64, n_pixels_dp: Int64,
):
    var fw = Int(fw_dp)
    var count = Int(count_dp)
    var n_pixels = Int(n_pixels_dp)
    var ti = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ti >= count:
        return
    var si_local = ti // n_pixels
    var px_flat  = ti % n_pixels
    var px = Int32(px_flat % fw)
    var py = Int32(px_flat // fw)
    var si = si_start + Int32(si_local)
    var rng_seed = UInt64(rng_seed_hi) << UInt64(32) | UInt64(rng_seed_lo)
    var (ray, pcg_state, pcg_inc, sobol_idx, wavelengths) = gen_primary_ray_state(
        px, py, si, Int(log2spp), Int(n_base4),
        seed_dim0, seed_dim1, rng_seed, sobol_matrices, r2c, c2w,
        filter_norm_x, filter_sigma, filter_support_x,
        filter_norm_y, filter_support_y,
        filter_type,
    )
    paths[unsafe_offset=ti] = PathState_C(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Float32(1.0),   # previous_dielectric_ior (vacuum)
        Float32(1.0),   # eta_scale
        Int32(3), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
        INV_FOUR_PI,    # lastEnvNeePdf (gated by lastBsdfPdf > 0; set at each scatter)
        Float32(0.0),   # cone_len: total path length, accumulated per bounce
    )


# Traversal kernel that reads rays directly from PathState_C (no separate ray buffer).
def traverse_paths_gpu(
    sd: SceneDescriptor2_C,
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    curve_cand_prim: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var n_spheres = Int(sd.sphereCount)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[unsafe_offset=tid].active == 0:
        return
    curve_cand_count[unsafe_offset=tid] = Int32(0)
    traverse_bvh2_core_defer_curves(
        sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, paths[unsafe_offset=tid].ray, Float32(1.0e38), results.unsafe_offset(tid),
        curve_cand_prim.unsafe_offset(tid * CURVE_DEFER_K), curve_cand_count.unsafe_offset(tid),
        sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
    )
    test_spheres(sd.spheres, n_spheres, paths[unsafe_offset=tid].ray, results.unsafe_offset(tid))


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
def vulkaninterop_pack_rays_kernel(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
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
    rays[unsafe_offset=idx + 7] = Float32(1.0e8)

# Unpack the interop-shared results buffer (written by Vulkan's ray-query
# dispatch) directly into inter_buf's Intersection_C layout -- no host
# copy. Same materialIndex/area-light lookup and PrimId_C.type==3 encoding
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
def vulkaninterop_unpack_results_kernel(
    results: Pointer[Float32, MutUntrackedOrigin],
    inter: Pointer[Intersection_C, MutUntrackedOrigin],
    mesh_material_idx: Pointer[Int64, MutUntrackedOrigin],
    mesh_al_idx: Pointer[Int32, MutUntrackedOrigin],
    n_meshes_dp: Int64,
    count_dp: Int64,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
):
    var n_meshes = Int(n_meshes_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid * 8
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
        var hitT = results[unsafe_offset=idx + 0]
        var u = results[unsafe_offset=idx + 1]
        var v = results[unsafe_offset=idx + 2]
        if al >= Int32(0):
            inter[unsafe_offset=tid] = Intersection_C(
                PrimId_C(Int64(al), (Int64(mi) << 32) | Int64(tri), mat_idx, Int32(-1),
                         Int8(3), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
        else:
            inter[unsafe_offset=tid] = Intersection_C(
                PrimId_C(Int64(mi), Int64(tri) * 3, mat_idx, instance_idx,
                         Int8(0), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
    elif hitFlag == Int32(2):
        # Curve hit, resolved directly by intersect_batch.comp itself (real
        # ray-vs-curve narrow-phase test + rayQueryGenerateIntersectionEXT
        # on a valid hit) -- no candidate buffer, no separate CUDA resolve
        # pass. hitMesh/hitTriangle/geometryIndex already carry PrimId_C's
        # id1 (curve index)/id2 (packed piece info)/materialIndex directly
        # (see vulkaninterop_rt_create_scene's docstring), so this is a
        # straight repack, no lookups needed. u/v here are intersect_curve's
        # own (h, v) outputs, not barycentrics.
        var curve_idx = Int64(iresults[unsafe_offset=idx + 4])
        var piece_info = Int64(iresults[unsafe_offset=idx + 5])
        var mat_idx = Int64(iresults[unsafe_offset=idx + 7])
        var hitT = results[unsafe_offset=idx + 0]
        var h = results[unsafe_offset=idx + 1]
        var v = results[unsafe_offset=idx + 2]
        inter[unsafe_offset=tid] = Intersection_C(
            PrimId_C(curve_idx, piece_info, mat_idx, Int32(-1), Int8(5), Int8(0), Int8(0), Int8(0)),
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
        var dummy_id = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
        inter[unsafe_offset=tid] = Intersection_C(dummy_id, Float32(1.0e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))

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
def vulkaninterop_test_spheres_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    inter: Pointer[Intersection_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
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
def accumulate_cone_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
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
    if results[unsafe_offset=tid].hit == Int8(0):
        return
    # ONLY along a specular chain. A ray cone tracks the CAMERA's footprint,
    # and that survives mirror reflection and refraction -- but at a diffuse
    # scatter the outgoing direction is random and the cone stops meaning
    # anything about the camera. Growing it there would blur deeper bounces
    # without bound and without justification; pbrt carries differentials for
    # camera rays and specular chains for the same reason. After the first
    # non-specular scatter the cone FREEZES at its last valid width, so those
    # bounces keep the footprint they legitimately had.
    if paths[unsafe_offset=tid].bounce == Int32(0) or paths[unsafe_offset=tid].specularBounce != Int8(0):
        paths[unsafe_offset=tid].cone_len += results[unsafe_offset=tid].tHit



def vulkaninterop_rt_traverse_paths_gpu(
    ctx: DeviceContext,
    path_buf: DeviceBuffer[DType.uint8],
    inter_buf: DeviceBuffer[DType.uint8],
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    interop_results_buf: DeviceBuffer[DType.float32],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin] = Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)

    ctx.enqueue_function[vulkaninterop_pack_rays_kernel](
        path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_total),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)

    var instance_base_mesh_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
    if instance_base_mesh_buf:
        instance_base_mesh_ptr = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()

    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        interop_results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_meshes),
        Int64(n_total),
        instance_base_mesh_ptr,
        grid_dim=grid, block_dim=block_size,
    )

    # Curves need no further handling here at all -- intersect_batch.comp
    # already resolved them (real narrow-phase test + generate on a valid
    # hit) as part of the SAME dispatch, and vulkaninterop_unpack_results_
    # kernel above already decoded a curve hit (hitFlag==2) into inter_buf.

    if n_spheres > 0:
        ctx.enqueue_function[vulkaninterop_test_spheres_gpu](
            path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
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

def resolve_curve_candidates_gpu(
    compact_pathIds: Pointer[Int32, MutUntrackedOrigin],
    compact_counter: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_prim: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_offset: Pointer[Int32, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
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
        results[unsafe_offset=pathId] = Intersection_C(best_prim, best_t, best_u, best_v, best_hit, 0, 0, 0)


# GPU kernel: generate primary PathState_C for every pixel in one pass.
# Each thread handles one pixel.  All sampling is pure math — no host calls.
def gen_primary_rays_gpu(
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count_dp: Int64,
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
    )
    paths[unsafe_offset=tid] = PathState_C(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Float32(1.0),   # previous_dielectric_ior (vacuum)
        Float32(1.0),   # eta_scale
        Int32(3), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
        INV_FOUR_PI,    # lastEnvNeePdf (gated by lastBsdfPdf > 0; set at each scatter)
        Float32(0.0),   # cone_len: total path length, accumulated per bounce
    )


# GPU kernel: shoot one unjittered center ray per pixel and write normals + depth.
# Used to guide the à-trous denoiser with geometric edge information.
def gen_aux_buffers_gpu(
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    isects_tmp: Pointer[Intersection_C, MutUntrackedOrigin],
    normals_out: Pointer[Float32, MutUntrackedOrigin],
    depth_out: Pointer[Float32, MutUntrackedOrigin],
    curve_mask_out: Pointer[Float32, MutUntrackedOrigin],
    world_pos_out: Pointer[Float32, MutUntrackedOrigin],
    material_id_out: Pointer[Int32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
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

    var ray = Ray_C(org, dir)
    var dummy_id = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    isects_tmp[unsafe_offset=tid] = Intersection_C(dummy_id, Float32(1e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
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
                var cu = _curve_perp_axis(ctangent)
                var cb = cross(ctangent, cu)
                var ch = isects_tmp[unsafe_offset=tid].u
                var cs = sqrt(max(Float32(0.0), Float32(1.0) - ch*ch))
                var cn = cu*ch + cb*cs
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
            var vi0 = Int(mesh.vertexIndices[unsafe_offset=base_vidx])
            var vi1 = Int(mesh.vertexIndices[unsafe_offset=base_vidx + 1])
            var vi2 = Int(mesh.vertexIndices[unsafe_offset=base_vidx + 2])
            var p0 = Point3f(mesh.points[unsafe_offset=vi0*4], mesh.points[unsafe_offset=vi0*4+1], mesh.points[unsafe_offset=vi0*4+2])
            var p1 = Point3f(mesh.points[unsafe_offset=vi1*4], mesh.points[unsafe_offset=vi1*4+1], mesh.points[unsafe_offset=vi1*4+2])
            var p2 = Point3f(mesh.points[unsafe_offset=vi2*4], mesh.points[unsafe_offset=vi2*4+1], mesh.points[unsafe_offset=vi2*4+2])
            var e1 = p1 - p0; var e2 = p2 - p0
            normal = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
            var inst_idx = isects_tmp[unsafe_offset=tid].primId.instanceIdx
            if inst_idx >= Int32(0):
                var n_world = transform_normal_by_instance(instances[unsafe_offset=Int(inst_idx)].worldToObj, normal.to_simd())
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
            handle[].ctx.enqueue_function[gen_aux_buffers_gpu](
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].bvh.nodes_ptr(),
                handle[].bvh.prim_ids_ptr(),
                handle[].meshes.meshes_ptr(),
                handle[].curves.curves_ptr(),
                handle[].blas.nodes_arr(),
                handle[].blas.primids_arr(),
                handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
                handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
                Int64(handle[].n_spheres),
                handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                handle[].atrous_normals_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_curve_mask_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].gbuf_material_id_buf.unsafe_ptr().unsafe_bitcast[Int32](),
                Int64(handle[].film.width), Int64(handle[].film.height),
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
# actual kernel bodies elsewhere in this codebase (see bdpt.mojo's MNEE/
# _connect duplication comments for that unrelated, still-real constraint).
def deactivate_paths_past_maxdepth_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
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
