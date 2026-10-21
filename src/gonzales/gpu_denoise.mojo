from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from .geometry import RGB, Vec3f, dot, cross
from std.math import sqrt, floor
from .postprocess import _atrous_spatial_weight, _atrous_tap_weight, _firefly_clamp_pixel
from max.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.sys import has_accelerator
from .gpu_scene import GpuSceneHandle


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def normalize_beauty_albedo_gpu(
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
    beauty_out: Pointer[Float32, MutUntrackedOrigin],
    albedo_out: Pointer[Float32, MutUntrackedOrigin],
    n_pixels_dp: Int64,
    inv_weight: Float32,
    iso_scale: Float32,
    max_comp: Float32,
):
    var n_pixels = Int(n_pixels_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    var lr = film[unsafe_offset=tid*3+0] * inv_weight * iso_scale
    var lg = film[unsafe_offset=tid*3+1] * inv_weight * iso_scale
    var lb = film[unsafe_offset=tid*3+2] * inv_weight * iso_scale
    # NaN guard only -- NOT a negative clamp; see apply_film_sensor's
    # docstring (rendering.mojo) for why a negative component here must
    # survive to the wb matrix multiply that follows this normalizer.
    if lr != lr: lr = Float32(0)
    if lg != lg: lg = Float32(0)
    if lb != lb: lb = Float32(0)
    var c = RGB(lr, lg, lb).sensor_clamped(max_comp)
    beauty_out[unsafe_offset=tid*3+0] = c.r
    beauty_out[unsafe_offset=tid*3+1] = c.g
    beauty_out[unsafe_offset=tid*3+2] = c.b
    albedo_out[unsafe_offset=tid*3+0] = albedo_film[unsafe_offset=tid*3+0] * inv_weight
    albedo_out[unsafe_offset=tid*3+1] = albedo_film[unsafe_offset=tid*3+1] * inv_weight
    albedo_out[unsafe_offset=tid*3+2] = albedo_film[unsafe_offset=tid*3+2] * inv_weight


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def estimate_variance_gpu(
    beauty: Pointer[Float32, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    film_sq: Pointer[Float32, MutUntrackedOrigin],
    variance_out: Pointer[Float32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    inv_weight: Float32, iso_scale: Float32, n_samples: Float32,
):
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw
    var py = tid // fw
    var mean = Float32(0)
    var mean_sq = Float32(0)
    var samp_var = Float32(0)
    var count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var nx = px + dx; var ny = py + dy
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            var ni = (ny * fw + nx) * 3
            var l = Float32(0.2126)*beauty[unsafe_offset=ni] + Float32(0.7152)*beauty[unsafe_offset=ni+1] + Float32(0.0722)*beauty[unsafe_offset=ni+2]
            mean += l; mean_sq += l * l; count += 1
            # Variance of the pixel's mean from its own samples: texture is not noise.
            var sl = (Float32(0.2126)*film[unsafe_offset=ni] + Float32(0.7152)*film[unsafe_offset=ni+1] + Float32(0.0722)*film[unsafe_offset=ni+2]) * inv_weight
            var sv = film_sq[unsafe_offset=ny * fw + nx] * inv_weight - sl * sl
            samp_var += (sv if sv > Float32(0) else Float32(0)) * iso_scale * iso_scale / n_samples
    var fc = Float32(count)
    mean /= fc; mean_sq /= fc; samp_var /= fc
    var v = mean_sq - mean * mean
    v = v if v > Float32(0) else Float32(0)
    # The 3x3 spatial estimate mistakes texture for noise; trust the sample estimate as samples accrue.
    var w = min(Float32(1), (n_samples - Float32(1)) / Float32(8))
    variance_out[unsafe_offset=tid] = w * samp_var + (Float32(1) - w) * v


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def firefly_clamp_gpu(
    beauty: Pointer[Float32, MutUntrackedOrigin],
    output: Pointer[Float32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
):
    """GPU counterpart of postprocess.mojo's _clamp_fireflies, which the GPU
    à-trous path never had until now -- a live divergence (see
    project_gpu_denoiser_energy_bug.md memory): without it, a single
    extreme-radiance pixel smears across the filter's full effective
    radius (up to 31px at 5 passes) exactly as it did on CPU before that
    fix existed. Same isolated-pixel test as CPU, via the SAME shared
    _firefly_clamp_pixel -- only the neighbor-gathering loop differs (one
    GPU thread per pixel vs a nested CPU loop)."""
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw
    var py = tid // fw
    var max_n = Float32(0)
    var max_n_r = Float32(0)
    var max_n_g = Float32(0)
    var max_n_b = Float32(0)
    var has_neighbor = False
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            var nx = px + dx
            var ny = py + dy
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            has_neighbor = True
            var ni = (ny * fw + nx) * 3
            var lum_n = RGB(beauty[unsafe_offset=ni], beauty[unsafe_offset=ni + 1], beauty[unsafe_offset=ni + 2]).luma()
            if lum_n > max_n:
                max_n = lum_n
            if beauty[unsafe_offset=ni + 0] > max_n_r: max_n_r = beauty[unsafe_offset=ni + 0]
            if beauty[unsafe_offset=ni + 1] > max_n_g: max_n_g = beauty[unsafe_offset=ni + 1]
            if beauty[unsafe_offset=ni + 2] > max_n_b: max_n_b = beauty[unsafe_offset=ni + 2]
    var ci = tid * 3
    var c = _firefly_clamp_pixel(
        beauty[unsafe_offset=ci + 0], beauty[unsafe_offset=ci + 1], beauty[unsafe_offset=ci + 2],
        max_n, max_n_r, max_n_g, max_n_b, has_neighbor)
    output[unsafe_offset=ci + 0] = c.r
    output[unsafe_offset=ci + 1] = c.g
    output[unsafe_offset=ci + 2] = c.b


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def temporal_accumulate_gpu(
    cur: Pointer[Float32, MutUntrackedOrigin],
    world_pos: Pointer[Float32, MutUntrackedOrigin],
    depth: Pointer[Float32, MutUntrackedOrigin],
    prev_depth: Pointer[Float32, MutUntrackedOrigin],
    hist: Pointer[Float32, MutUntrackedOrigin],
    hist_len: Pointer[Float32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    prev_c2w: Pointer[Float32, MutUntrackedOrigin],
    result: Pointer[Float32, MutUntrackedOrigin],
    len_out: Pointer[Float32, MutUntrackedOrigin],
    fw_i32: Int32, fh_i32: Int32, n_samples: Float32,
):
    """Blend the spatially denoised frame with last frame's output, reprojected through the
    previous camera by the first-hit world position. Taps whose previous depth disagrees are dropped."""
    var fw = Int(fw_i32); var fh = Int(fh_i32)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw; var py = tid // fw
    var c = RGB(cur[unsafe_offset=tid*3], cur[unsafe_offset=tid*3+1], cur[unsafe_offset=tid*3+2])
    result[unsafe_offset=tid*3] = c.r; result[unsafe_offset=tid*3+1] = c.g; result[unsafe_offset=tid*3+2] = c.b
    len_out[unsafe_offset=tid] = Float32(1)
    if depth[unsafe_offset=tid] > Float32(1e30):
        return
    var wx = world_pos[unsafe_offset=tid*3] - prev_c2w[unsafe_offset=12]
    var wy = world_pos[unsafe_offset=tid*3+1] - prev_c2w[unsafe_offset=13]
    var wz = world_pos[unsafe_offset=tid*3+2] - prev_c2w[unsafe_offset=14]
    var dist = sqrt(wx*wx + wy*wy + wz*wz)
    var P = Vec3f(wx*prev_c2w[unsafe_offset=0] + wy*prev_c2w[unsafe_offset=1] + wz*prev_c2w[unsafe_offset=2],
                  wx*prev_c2w[unsafe_offset=4] + wy*prev_c2w[unsafe_offset=5] + wz*prev_c2w[unsafe_offset=6],
                  wx*prev_c2w[unsafe_offset=8] + wy*prev_c2w[unsafe_offset=9] + wz*prev_c2w[unsafe_offset=10])
    if P.z <= Float32(0):
        return
    # Raster -> camera is linear in (x, y, 1) up to a positive scale; invert it for P.
    var c0 = Vec3f(r2c[unsafe_offset=0], r2c[unsafe_offset=1], r2c[unsafe_offset=2])
    var c1 = Vec3f(r2c[unsafe_offset=4], r2c[unsafe_offset=5], r2c[unsafe_offset=6])
    var c2 = Vec3f(r2c[unsafe_offset=12], r2c[unsafe_offset=13], r2c[unsafe_offset=14])
    var det = dot(c0, cross(c1, c2))
    var u2 = dot(c0, cross(c1, P)) / det
    if abs(u2) < Float32(1e-20):
        return
    var rx = dot(P, cross(c1, c2)) / det / u2 - Float32(0.5)
    var ry = dot(c0, cross(P, c2)) / det / u2 - Float32(0.5)
    var x0 = Int(floor(rx)); var y0 = Int(floor(ry))
    var fx = rx - Float32(x0); var fy = ry - Float32(y0)
    # Grazing surfaces change depth fast from pixel to pixel: widen the tolerance by the local slope.
    var d0 = depth[unsafe_offset=tid]
    var slope = Float32(0)
    if px + 1 < fw and depth[unsafe_offset=tid + 1] < Float32(1e30):
        slope = abs(depth[unsafe_offset=tid + 1] - d0)
    if py + 1 < fh and depth[unsafe_offset=tid + fw] < Float32(1e30):
        slope = max(slope, abs(depth[unsafe_offset=tid + fw] - d0))
    var tol = Float32(0.03) * dist + Float32(2) * slope
    var acc = RGB(Float32(0)); var acc_len = Float32(0); var acc_w = Float32(0)
    for j in range(2):
        for i in range(2):
            var nx = x0 + i; var ny = y0 + j
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            var ni = ny * fw + nx
            if abs(prev_depth[unsafe_offset=ni] - dist) > tol:
                continue
            var w = (fx if i == 1 else Float32(1) - fx) * (fy if j == 1 else Float32(1) - fy)
            acc += RGB(hist[unsafe_offset=ni*3], hist[unsafe_offset=ni*3+1], hist[unsafe_offset=ni*3+2]) * w
            acc_len += hist_len[unsafe_offset=ni] * w
            acc_w += w
    if acc_w < Float32(0.05):
        return
    var h = acc / acc_w
    var len_new = min(acc_len / acc_w + Float32(1), Float32(24))
    var a = max(Float32(1) / len_new, min(Float32(1), n_samples / Float32(32)))
    var o = h * (Float32(1) - a) + c * a
    result[unsafe_offset=tid*3] = o.r; result[unsafe_offset=tid*3+1] = o.g; result[unsafe_offset=tid*3+2] = o.b
    len_out[unsafe_offset=tid] = len_new


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def atrous_filter_gpu(
    input: Pointer[Float32, MutUntrackedOrigin],
    albedo: Pointer[Float32, MutUntrackedOrigin],
    variance: Pointer[Float32, MutUntrackedOrigin],
    normals: Pointer[Float32, MutUntrackedOrigin],
    depth: Pointer[Float32, MutUntrackedOrigin],
    curve_mask: Pointer[Float32, MutUntrackedOrigin],
    output: Pointer[Float32, MutUntrackedOrigin],
    fw_i32: Int32, fh_i32: Int32,
    step_i32: Int32,
    sigma_l: Float32,
    sigma_a: Float32,
    sigma_n: Float32,
    sigma_d: Float32,
):
    var fw = Int(fw_i32); var fh = Int(fh_i32); var step = Int(step_i32)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw; var py = tid // fw

    var c = RGB(input[unsafe_offset=tid*3], input[unsafe_offset=tid*3+1], input[unsafe_offset=tid*3+2])
    if curve_mask[unsafe_offset=tid] > Float32(0.5):
        # Hair/fur: strand-to-strand self-shadowing has no reliable correlate in
        # albedo/normal/depth (adjacent strands share material and similar
        # orientation/distance), so à-trous can't tell real occlusion from noise
        # and blurs it into a flat blob. Passing raw beauty through here matches
        # pbrt's own un-denoised look for hair instead of erasing strand detail.
        output[unsafe_offset=tid*3] = c.r; output[unsafe_offset=tid*3+1] = c.g; output[unsafe_offset=tid*3+2] = c.b
        return
    var cl = c.luma()
    var var_p = variance[unsafe_offset=tid]
    var ca = RGB(albedo[unsafe_offset=tid*3], albedo[unsafe_offset=tid*3+1], albedo[unsafe_offset=tid*3+2])
    var cn = Vec3f(normals[unsafe_offset=tid*3], normals[unsafe_offset=tid*3+1], normals[unsafe_offset=tid*3+2])
    # Clamp depth before squaring to avoid Float32 overflow (background sentinel=1e38).
    var cd_clamped = min(depth[unsafe_offset=tid], Float32(1e18))
    var cd_sq = max(cd_clamped * cd_clamped, Float32(1e-6))

    var acc = RGB(Float32(0))
    var acc_w = Float32(0)

    # Per-tap weight is _atrous_tap_weight (postprocess.mojo), shared
    # VERBATIM with the CPU denoise() pass loop -- see that function's
    # own docstring for why min(var_p,var_q), not var_p alone, matters
    # (an asymmetric weight destroys energy instead of moving it; this
    # was a real, measured bug -- project_gpu_denoiser_energy_bug.md).
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            var nx = px + dx * step; var ny = py + dy * step
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            var ni = (ny * fw + nx) * 3
            var ni1 = ny * fw + nx
            if curve_mask[unsafe_offset=ni1] > Float32(0.5):
                continue
            var qc = RGB(input[unsafe_offset=ni], input[unsafe_offset=ni+1], input[unsafe_offset=ni+2])
            var dl = qc.luma() - cl
            var dalb = RGB(albedo[unsafe_offset=ni], albedo[unsafe_offset=ni+1], albedo[unsafe_offset=ni+2]) - ca
            var ndot = normals[unsafe_offset=ni]*cn.x + normals[unsafe_offset=ni+1]*cn.y + normals[unsafe_offset=ni+2]*cn.z
            var dd = min(depth[unsafe_offset=ni1], Float32(1e18)) - cd_clamped
            var w = _atrous_spatial_weight(dx, dy) * _atrous_tap_weight(
                dl, var_p, variance[unsafe_offset=ni1], dalb, ndot, dd, cd_sq,
                sigma_l, sigma_a, sigma_n, sigma_d)
            acc += qc * w
            acc_w += w

    if acc_w > Float32(0):
        var o = acc / acc_w
        output[unsafe_offset=tid*3] = o.r; output[unsafe_offset=tid*3+1] = o.g; output[unsafe_offset=tid*3+2] = o.b
    else:
        output[unsafe_offset=tid*3] = c.r; output[unsafe_offset=tid*3+1] = c.g; output[unsafe_offset=tid*3+2] = c.b


def gpu_atrous_denoise[Oo: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    output: Pointer[Float32, Oo],
    n: Int64,
    frame_count: Int32,
    film_iso: Float32,
    film_max_comp: Float32,
    apply_denoise: Bool = True,
    min_passes: Int = 1,    # the interactive viewer wants the full filter at once, also right after a camera move
):
    var n_pix = Int(n)
    if n_pix == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            var fw = Int(handle[].film.width); var fh = Int(handle[].film.height)
            comptime block_size = 256
            var grid_n = ceildiv(n_pix, block_size)
            var inv_weight = Float32(1.0) / Float32(max(Int(frame_count), 1))
            var iso_scale = film_iso / Float32(100.0)

            handle[].ctx.enqueue_function[normalize_beauty_albedo_gpu](
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_ping_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_albedo_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_pix), inv_weight, iso_scale, film_max_comp,
                grid_dim=grid_n, block_dim=block_size,
            )
            # --no-denoise: emit the normalized beauty (atrous_ping_buf) without
            # the à-trous blur passes, so the written image is the raw render.
            if not apply_denoise:
                handle[].ctx.enqueue_copy(output.unsafe_bitcast[UInt8](), handle[].atrous_ping_buf)
                handle[].ctx.synchronize()
                return
            # Firefly pre-clamp -- matches CPU's denoise() (postprocess.mojo),
            # which GPU never had before this. Without it a single extreme
            # pixel smears across the filter's full effective radius (up to
            # 31px at 5 passes). Writes into atrous_pong_buf: pass 0 below
            # then reads from THAT (clamped) buffer, reusing atrous_ping_buf
            # (whose unclamped contents are no longer needed) as scratch --
            # ping/pong roles are therefore swapped relative to before this
            # change, tracked explicitly via clamp_dst_ptr below rather than
            # implicitly through the i%2 alternation.
            handle[].ctx.enqueue_function[firefly_clamp_gpu](
                handle[].atrous_ping_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_pong_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(fw), Int64(fh),
                grid_dim=grid_n, block_dim=block_size,
            )
            var clamp_dst_ptr = handle[].atrous_pong_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            handle[].ctx.enqueue_function[estimate_variance_gpu](
                clamp_dst_ptr,
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].film_sq_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_variance_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(fw), Int64(fh),
                inv_weight, iso_scale, Float32(max(Int(frame_count), 1)),
                grid_dim=grid_n, block_dim=block_size,
            )

            var ping_ptr = handle[].atrous_ping_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var pong_ptr = handle[].atrous_pong_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var alb_ptr  = handle[].atrous_albedo_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var var_ptr  = handle[].atrous_variance_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var nrm_ptr  = handle[].atrous_normals_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var dep_ptr  = handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var cmask_ptr = handle[].atrous_curve_mask_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            # Ramp passes with frame_count: 1 pass at fc=1, 5 passes at fc>=5.
            # Prevents the large effective radius (31px at 5 passes) from averaging
            # lit pixels with unlit ones during fast camera movement.
            var n_passes = min(5, max(min_passes, Int(frame_count)))
            # Narrower luminance tolerance, halved per pass: 4.0 flattened foliage texture into paint.
            var sl = Float32(2.0)
            for i in range(n_passes):
                var step = 1 << i   # 1, 2, 4, 8, 16
                # Pass 0 reads the CLAMPED buffer (pong), not ping -- see the
                # firefly-clamp comment above for why the starting side is
                # swapped from the pre-firefly-clamp version of this loop.
                var src_ptr = pong_ptr if i % 2 == 0 else ping_ptr
                var dst_ptr = ping_ptr if i % 2 == 0 else pong_ptr
                handle[].ctx.enqueue_function[atrous_filter_gpu](
                    src_ptr, alb_ptr, var_ptr, nrm_ptr, dep_ptr, cmask_ptr, dst_ptr,
                    Int32(fw), Int32(fh), Int32(step),
                    sl, Float32(0.1), Float32(0.3), Float32(0.05),
                    grid_dim=grid_n, block_dim=block_size,
                )
                sl *= Float32(0.5)
            # Result is in ping if n_passes is odd, pong if even (start=pong); the other one is free.
            var res_ptr = ping_ptr if n_passes % 2 == 1 else pong_ptr
            var free_ptr = pong_ptr if n_passes % 2 == 1 else ping_ptr
            handle[].ctx.enqueue_function[temporal_accumulate_gpu](
                res_ptr,
                handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                dep_ptr,
                handle[].prev_depth_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].hist_color_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].hist_len_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].prev_c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                free_ptr,
                handle[].hist_len_tmp_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int32(fw), Int32(fh), Float32(max(Int(frame_count), 1)),
                grid_dim=grid_n, block_dim=block_size,
            )
            handle[].ctx.enqueue_copy(handle[].hist_color_buf, handle[].atrous_pong_buf if n_passes % 2 == 1 else handle[].atrous_ping_buf)
            handle[].ctx.enqueue_copy(handle[].hist_len_buf, handle[].hist_len_tmp_buf)
            handle[].ctx.enqueue_copy(handle[].prev_depth_buf, handle[].atrous_depth_buf)
            handle[].ctx.enqueue_copy(handle[].prev_c2w_buf, handle[].c2w_buf)
            handle[].ctx.enqueue_copy(output.unsafe_bitcast[UInt8](), handle[].atrous_pong_buf if n_passes % 2 == 1 else handle[].atrous_ping_buf)
            handle[].ctx.synchronize()
        except e:
            print("GPU atrous denoise failed: " + String(e))
