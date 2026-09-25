from .geometry import RGB, Vec3f
from .postprocess import _atrous_spatial_weight, _atrous_tap_weight, _firefly_clamp_pixel
from max.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.sys import has_accelerator
from .gpu_scene import GpuSceneHandle


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
    if lr != lr or lr < Float32(0): lr = Float32(0)
    if lg != lg or lg < Float32(0): lg = Float32(0)
    if lb != lb or lb < Float32(0): lb = Float32(0)
    var scale = Float32(1.0)
    if max_comp > Float32(0.0):
        var mx = lr if lr > lg else lg
        if lb > mx: mx = lb
        if mx > max_comp:
            scale = max_comp / mx
    beauty_out[unsafe_offset=tid*3+0] = lr * scale
    beauty_out[unsafe_offset=tid*3+1] = lg * scale
    beauty_out[unsafe_offset=tid*3+2] = lb * scale
    albedo_out[unsafe_offset=tid*3+0] = albedo_film[unsafe_offset=tid*3+0] * inv_weight
    albedo_out[unsafe_offset=tid*3+1] = albedo_film[unsafe_offset=tid*3+1] * inv_weight
    albedo_out[unsafe_offset=tid*3+2] = albedo_film[unsafe_offset=tid*3+2] * inv_weight


def estimate_variance_gpu(
    beauty: Pointer[Float32, MutUntrackedOrigin],
    variance_out: Pointer[Float32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
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
    var count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var nx = px + dx; var ny = py + dy
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            var ni = (ny * fw + nx) * 3
            var l = Float32(0.2126)*beauty[unsafe_offset=ni] + Float32(0.7152)*beauty[unsafe_offset=ni+1] + Float32(0.0722)*beauty[unsafe_offset=ni+2]
            mean += l; mean_sq += l * l; count += 1
    var fc = Float32(count)
    mean /= fc; mean_sq /= fc
    var v = mean_sq - mean * mean
    variance_out[unsafe_offset=tid] = v if v > Float32(0) else Float32(0)


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
                handle[].atrous_variance_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(fw), Int64(fh),
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
            var n_passes = min(5, max(1, Int(frame_count)))
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
                    Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05),
                    grid_dim=grid_n, block_dim=block_size,
                )
            # Result is in ping if n_passes is odd, pong if even (start=pong).
            if n_passes % 2 == 1:
                handle[].ctx.enqueue_copy(output.unsafe_bitcast[UInt8](), handle[].atrous_ping_buf)
            else:
                handle[].ctx.enqueue_copy(output.unsafe_bitcast[UInt8](), handle[].atrous_pong_buf)
            handle[].ctx.synchronize()
        except e:
            print("GPU atrous denoise failed: " + String(e))
