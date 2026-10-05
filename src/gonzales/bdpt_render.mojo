# BDPT/VCM CPU render driver.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.sys.info import size_of
from max.algorithm import parallelize
from std.math import tan, max, abs
from std.memory.alloc import unsafe_alloc
from .geometry import RGB, Vec3f, PI
from .primitives import Intersection
from .vcm_camis import CamisLightRecord
from .bvh import SceneView, _scene_bounding_sphere
from .sampling import film_filter_of
from .rng import PCG32
from .transform import matrix_invert
from .pbrt_parser import ParsedScene_Mojo
from .progress import Progress
from .outputs import finish_render
from .spectrum import SpectralSample, pass_wavelengths, spectral_sample_to_rgb
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .vcm_grid import (
    _VCM_MN_STRIDE, _VCM_HEADS_SIZE, _CAMIS_LVC_PER_SLOT, _vcm_grid_inv_cell, _bdpt_build_merge_grid,
    vcm_merge_radius,
)
from .bdpt_connect import _bdpt_splat_filtered, _bdpt_connect_to_camera
from .bdpt_camera import _bdpt_trace_camera_and_connect
from .bdpt_light import _bdpt_trace_light_path

def _vcm_finalize_one_pixel(
    connect: RGB, merge: RGB, inv_spp: Float32,
) -> Tuple[RGB, RGB]:
    """Averages one pixel's accumulated connect+splat and vertex-merging
    sums over the render's spp samples, returning them SEPARATELY --
    mirrors sppm.mojo's `_sppm_finalize_one_pixel` (same reasoning: see
    VCMCameraPathState.total_merge's docstring and
    project_water_caustic_sppm_gap memory). NaN/negative-guarded per
    component, order-independent and safe to do before the caller sums the
    two halves; does NOT apply the max-component sensor clamp -- same as
    SPPM's finalize, that happens ONCE in finish_render after `connect` is
    denoised and `merge` is added back unsmoothed (clamping each half
    independently first would under-clamp their sum on scenes that set
    `maxcomponentvalue`). Shared by every VCM driver (CPU `_bdpt_render_core`,
    GPU `_bdpt_camera_connect_gpu`, GPU-wavefront's own tail) so the split
    lives in one place instead of three."""
    var c = connect * inv_spp
    var m = merge * inv_spp
    if c.r != c.r or c.r < Float32(0): c.r = Float32(0)
    if c.g != c.g or c.g < Float32(0): c.g = Float32(0)
    if c.b != c.b or c.b < Float32(0): c.b = Float32(0)
    if m.r != m.r or m.r < Float32(0): m.r = Float32(0)
    if m.g != m.g or m.g < Float32(0): m.g = Float32(0)
    if m.b != m.b or m.b < Float32(0): m.b = Float32(0)
    return (c, m)


def _bdpt_render_core(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_spp:    Int,
    n_photons_req: Int,
    verbose:  Bool,
) -> Tuple[Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]]:
    """Bidirectional Path Tracing main loop with real VCM connect+merge MIS
    (Light Vertex Cache architecture — see the module docstring above),
    factored out of `vcm_render` (its CLI-facing caller, below) so the CPU
    and GPU entry points share the exact same core. Returns (pixels,
    caustic_pixels, albedo_pixels), each a caller-owned `n_pix*3` Float32
    buffer, iso-scaled but NOT max_comp-clamped and NOT yet denoised --
    same contract `_sppm_render_core` follows, and for the same reason:
    `pixels` (connect + t=1 splats) and `caustic_pixels` (vertex merging)
    are kept separate so the caller can denoise only `pixels` and add
    `caustic_pixels` back unsmoothed (see _vcm_finalize_one_pixel's
    docstring). The clamp happens once, in finish_render, on their sum.

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
    # `buf` holds connect + t=1 light-tracing splats (ordinary noise
    # character); `buf_merge` holds vertex-MERGING contributions only,
    # split out for the SAME reason SPPM's finalize splits its gather term
    # -- see VCMCameraPathState.total_merge's docstring.
    var buf = unsafe_alloc[RGB](n_pix)
    var buf_merge = unsafe_alloc[RGB](n_pix)
    var albedo_buf = unsafe_alloc[RGB](n_pix)
    for i in range(n_pix):
        buf[unsafe_offset=i] = RGB(Float32(0))
        buf_merge[unsafe_offset=i] = RGB(Float32(0))
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
    # One scratch Intersection per concurrent worker (light path / pixel)
    # instead of one shared slot — CPU threads now race on this exactly like
    # GPU threads already do (see _bdpt_emit_light_paths_gpu/
    # _bdpt_camera_connect_gpu's own per-thread inter_light_ptr+k/
    # inter_cam_ptr+pix), so it can no longer be a single reused buffer.
    var scratch_light = unsafe_alloc[Intersection](max(n_light_paths_merge, 1))
    var scratch_cam = unsafe_alloc[Intersection](max(n_pix, 1))

    # VCM vertex merging: grid buffers allocated once, rebuilt fresh every
    # spp sample (mirrors the LVC itself). Stage 2c: the radius itself is
    # now progressive (Hachisuka & Jensen 2008's global, per-iteration
    # scheme -- see this file's opening VCM comment for why that's the
    # right choice here, not SPPM's per-pixel adaptive radius), recomputed
    # each `si` below from `merge_radius_1`, the same initial 3%-of-scene-
    # diameter value Stage 1 used as its (then-fixed) radius.
    var (_scene_center, scene_radius) = _scene_bounding_sphere(sd)
    var merge_heads = unsafe_alloc[Int32](_VCM_HEADS_SIZE)   # heads | counts | fine levels, see _VCM_FINE_LEVELS
    var merge_next = unsafe_alloc[Int32](max(lvc_cap, 1) * _VCM_MN_STRIDE)
    # CAMIS light records (vcm_camis.CamisLightRecord), slot for slot with
    # `lvc`. A single dummy element when the hybrid is compiled out.
    var lvc_camis = unsafe_alloc[CamisLightRecord](max(lvc_cap * _CAMIS_LVC_PER_SLOT, 1))
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

    var prog = Progress(n_spp, "spp", quiet=verbose)
    for si in range(n_spp):
        # Stage 2c progressive radius: r_i = r_1 / (i+1)^(0.5*(1-alpha))
        # (Hachisuka & Jensen 2008 via Georgiev et al. 2012 Eq. 11), a
        # single GLOBAL radius shared by every pixel this sample, shrinking
        # monotonically across samples -- distinct from sppm.mojo's
        # per-pixel Knaus-Zwicker scheme (see this file's opening comment).
        var radius_i = vcm_merge_radius(scene_radius, si)
        var merge_r2 = radius_i * radius_i
        var merge_inv_cell = _vcm_grid_inv_cell(sd, radius_i)
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
        def emit_light_path(lp_idx: Int) {imm}:
            var lpcg = PCG32(base_seed ^ UInt64(lp_idx * 6364136223846793005 + 1442695040888963407),
                              UInt64(si * 2654435761 + 1))
            _bdpt_trace_light_path[False](sd, lpcg, has_med, default_emit_med,
                                         scratch_light.unsafe_offset(lp_idx), lvc, lp_idx, lvc_path_len,
                                         mis_vc_weight_factor, mis_vm_weight_factor, pass_wl, lvc_camis)

        parallelize(emit_light_path, n_light_paths_merge)

        _bdpt_build_merge_grid(lvc, lvc_path_len, n_light_paths_merge, merge_next, merge_heads, merge_inv_cell, sd)

        # ── Phase 1.5: t=1 light tracing (splat) ─────────────────────────
        # A light vertex lands on an ARBITRARY pixel, not the one being
        # shaded, so this cannot be folded into Phase 2's per-pixel
        # accumulation the way every other strategy is. Rather than add
        # float atomics on the film, the connections (which do the
        # expensive part -- a visibility ray each) run in parallel into a
        # per-slot record array, and the cheap accumulation is then a
        # serial pass. That also keeps the result deterministic, which
        # atomics on floats would not.
        def splat_light_path(lp_idx: Int) {imm}:
            var base = lp_idx * _BDPT_MAX_VERTS
            for local in range(Int(lvc_path_len[unsafe_offset=lp_idx])):
                var r = _bdpt_connect_to_camera(
                    lvc[unsafe_offset=base + local], sd, scratch_light.unsafe_offset(lp_idx), cam_pos,
                    w2c, c2r, Int32(fw), Int32(fh), px_scale,
                    Float32(n_light_paths_merge), mis_vm_weight_factor,
                    lvc, lvc_camis, base + local)
                splat_fx[unsafe_offset=base + local] = r[1] if r[0] else Float32(-1)
                splat_fy[unsafe_offset=base + local] = r[2]
                splat_val[unsafe_offset=base + local] = r[3]
            for local in range(Int(lvc_path_len[unsafe_offset=lp_idx]), _BDPT_MAX_VERTS):
                splat_fx[unsafe_offset=base + local] = Float32(-1)

        parallelize(splat_light_path, n_light_paths_merge)

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
        def camera_connect(pix: Int) {imm}:
            var px = pix % fw; var py = pix // fw
            var cpcg = PCG32(base_seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                              UInt64(si * 2654435761 + 1))
            var (contrib, contrib_merge, alb, _cpu_vn, _cpu_vf, _cpu_vt) = _bdpt_trace_camera_and_connect[False](
                r2c, c2w, px, py, sd, cpcg, has_med, scratch_cam.unsafe_offset(pix), lvc, pix, Int(lvc_path_len[unsafe_offset=pix]),
                merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
                px_scale, mis_vc_weight_factor, mis_vm_weight_factor, Float32(n_light_paths_merge), pass_wl,
                film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                               psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                lvc_camis=lvc_camis)
            # ── Output boundary: spectral transport -> RGB film ──────────
            var (cr, cg, cb) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, contrib, pass_wl)
            buf[unsafe_offset=pix] += RGB(cr, cg, cb)
            var (mr, mg, mb) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, contrib_merge, pass_wl)
            buf_merge[unsafe_offset=pix] += RGB(mr, mg, mb)
            albedo_buf[unsafe_offset=pix] += alb

        parallelize(camera_connect, n_pix)

        if verbose:
            print("VCM: sample " + String(si + 1) + "/" + String(n_spp))
        prog.update(si + 1)
    _ = prog.finish()

    scratch_light.unsafe_free(); scratch_cam.unsafe_free(); lvc.unsafe_free(); lvc_path_len.unsafe_free()
    merge_heads.unsafe_free(); merge_next.unsafe_free(); lvc_camis.unsafe_free()
    # Were never freed (the old splat_pix leaked the same way), once per render.
    splat_fx.unsafe_free(); splat_fy.unsafe_free(); splat_val.unsafe_free()

    # Split into caller-owned output buffers (no clamp, no denoise/write
    # here -- see vcm_render/vcm_render_gpu, this function's two callers,
    # for the tail; the max-component clamp now happens ONCE in
    # finish_render, on connect+merge summed together, after connect alone
    # is denoised -- see _vcm_finalize_one_pixel's docstring).
    var inv_spp = iso_scale / Float32(n_spp)
    var pixels = unsafe_alloc[Float32](n_pix * 3)
    var caustic_pixels = unsafe_alloc[Float32](n_pix * 3)
    for i in range(n_pix):
        var (c, m) = _vcm_finalize_one_pixel(buf[unsafe_offset=i], buf_merge[unsafe_offset=i], inv_spp)
        pixels[unsafe_offset=i*3]   = c.r
        pixels[unsafe_offset=i*3+1] = c.g
        pixels[unsafe_offset=i*3+2] = c.b
        caustic_pixels[unsafe_offset=i*3]   = m.r
        caustic_pixels[unsafe_offset=i*3+1] = m.g
        caustic_pixels[unsafe_offset=i*3+2] = m.b
    buf.unsafe_free(); buf_merge.unsafe_free()

    var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
    var inv_spp_alb = Float32(1) / Float32(n_spp)
    for i in range(n_pix):
        var a = albedo_buf[unsafe_offset=i] * inv_spp_alb
        albedo_pixels[unsafe_offset=i*3]   = a.r
        albedo_pixels[unsafe_offset=i*3+1] = a.g
        albedo_pixels[unsafe_offset=i*3+2] = a.b
    albedo_buf.unsafe_free()

    return Tuple[Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](pixels, caustic_pixels, albedo_pixels)

def vcm_render(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
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
    var (pixels, caustic_pixels, albedo_pixels) = _bdpt_render_core(psc, sd, n_spp, n_photons, verbose)

    _ = finish_render(psc, sd, pixels, albedo_pixels, no_denoise, caustic_pixels)
    pixels.unsafe_free(); caustic_pixels.unsafe_free(); albedo_pixels.unsafe_free()
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
# Every kernel takes the scene as one `sd: SceneView`, built on the
# host by GpuSceneHandle.scene_descriptor() (+ with_vcm() for a VCM pass).
