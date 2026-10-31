"""SPPM on the CPU: its passes and their threads."""
from max.algorithm import parallelize
from std.math import max, min
from std.memory.alloc import unsafe_alloc
from gonzales.transport.medium_interface import default_emit_medium
from gonzales.scene.shapes.primitives import Intersection
from gonzales.scene.appearance.media import SSS_WALK_ROUNDS
from gonzales.scene.accel.bvh import SceneView
from gonzales.estimators.photon_grid import _HSIZE, grid_reset_cell
from gonzales.core.sampling import FilmFilter, film_filter_of
from gonzales.core.rng import PCG32
from gonzales.scene.description.pbrt_parser import ParsedScene_Mojo
from gonzales.core.progress import Progress
from gonzales.integrators.outputs import finish_render
from gonzales.core.spectrum import SampledWavelengths, pass_wavelengths
from gonzales.integrators.sppm.sppm import _photon_depth_cap, _VP_SAMPLES, SPPMPixel, SPPMPhoton, _sppm_trace_visible_point, _sppm_trace_photon, _sppm_count_photon, _sppm_insert_photon, _sppm_gather_one, _sppm_nee_one, _sppm_flush_one, _sppm_finalize_albedo_one_pixel, _sppm_finalize_one_pixel


def _sppm_camera_pass(
    vps:        Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix:      Int,
    vp_samples: Int,
    fw:       Int32,
    r2c:      Pointer[Float32, MutUntrackedOrigin],
    c2w:      Pointer[Float32, MutUntrackedOrigin],
    ref sd:       SceneView,
    init_r2:  Float32,
    seed:     UInt64,
    maxdepth: Int,
    film_filter: FilmFilter,
):
    """Trace `vp_samples` independent primary rays per pixel, ONCE for the
    whole render (not once per SPPM pass — see sppm_render's docstring for
    why a per-pass re-trace breaks SPPM's convergence guarantee). Each sample
    gets its own persistent (r2, tau, N_acc) accumulator, fixed at this one
    surface/position for every subsequent photon pass; sppm_render averages
    the `vp_samples` independently-converged estimates per pixel at the end.
    Having multiple independent samples (rather than one, traced once) is
    what avoids the old black-speckle bug: a pixel where the camera ray
    crosses a dielectric surface makes a genuine random reflect-vs-refract
    choice each sample, so as long as vp_samples is large enough, at least
    some samples land on the diffuse floor even if others reflect away."""
    # One scratch Intersection per worker (indexed by `combined`) instead
    # of one shared slot — same convention the GPU kernel already uses
    # (inter_scratch + combined, one per thread) — needed now that this loop
    # runs across CPU threads too, not just GPU ones.
    var scratch = unsafe_alloc[Intersection](max(n_pix * vp_samples, 1))

    def trace_one(combined: Int) {imm}:
        var pix = combined // vp_samples
        var px = pix % Int(fw)
        var py = pix // Int(fw)
        var pcg = PCG32(seed ^ UInt64(combined * 6364136223846793005 + 1), UInt64(1))
        vps[unsafe_offset=combined] = _sppm_trace_visible_point[False](sd, pcg, r2c, c2w, px, py, Int32(pix), init_r2, scratch.unsafe_offset(combined), maxdepth, film_filter, combined % vp_samples)

    parallelize(trace_one, n_pix * vp_samples)

    scratch.unsafe_free()


# ── Photon pass ───────────────────────────────────────────────────────────────


@always_inline
def _sppm_photon_pass(
    photons:      Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_emit:       Int,
    max_photons:  Int,
    ref sd:           SceneView,
    seed:         UInt64,
    pass_idx:     Int,
    maxdepth:     Int,
) -> Int:
    """CPU driver: emit n_emit photon paths, returning the number actually
    stored (clamped to max_photons)."""
    var n_lights = Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount)
    if n_lights == 0:
        return 0

    # One scratch Intersection per worker (indexed by k), same convention
    # as the GPU kernel's inter_scratch + k — needed now that this loop runs
    # across CPU threads too.
    var scratch = unsafe_alloc[Intersection](max(n_emit, 1))
    var counter = unsafe_alloc[Int32](1)
    counter[unsafe_offset=0] = Int32(0)

    # Determine the "default" starting medium for photons emitted into a medium.
    # Convention: photon is cosine-sampled from ln, so dot(pdir,ln)>0 always.
    # We use the first MediumInterface whose outside_medium_idx is valid.
    var default_emit_med = default_emit_medium(sd)

    # use_gpu=True: parallel CPU workers now need the SAME atomic slot-
    # reservation _sppm_store_photon uses for GPU threads (concurrent
    # racing writers to the shared `photons` buffer/`counter`), regardless
    # of which backend is actually running. tex_gpu=False: this driver IS
    # actually CPU, so _tex_lookup must use the CPU tex_filenames path
    # (see _sppm_trace_photon's docstring for why these are separate params).
    def emit_one(k: Int) {imm}:
        var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
        _sppm_trace_photon[True, False](sd, pcg, scratch.unsafe_offset(k), n_emit, photons, max_photons, counter, default_emit_med, maxdepth,
            sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
            sd.spectral.cie_z, sd.spectral.d65, pass_wavelengths(pass_idx * n_emit + k))

    parallelize(emit_one, n_emit)

    var n_stored = min(Int(counter[unsafe_offset=0]), max_photons)
    counter.unsafe_free()
    scratch.unsafe_free()
    return n_stored


# ── Hash grid ─────────────────────────────────────────────────────────────────


def _build_grid(
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_phot:   Int,
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    pass_idx: Int,
):
    def reset_one(i: Int) {imm}:
        grid_reset_cell(heads, i)

    parallelize(reset_one, _HSIZE)

    def count_one(k: Int) {imm}:
        _sppm_count_photon(k, photons, heads, inv_cell)

    parallelize(count_one, n_phot)

    # [True]: parallel CPU workers race on the same bucket heads a GPU
    # kernel's threads would, so need the same atomic-exchange insert.
    def insert_one(k: Int) {imm}:
        _sppm_insert_photon[True](k, photons, heads, inv_cell, pass_idx)

    parallelize(insert_one, n_phot)


# ── Gather + SPPM update ──────────────────────────────────────────────────────


def _gather_update(
    vps:      Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix:    Int,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    ref sd:       SceneView,
    pass_wl:  SampledWavelengths,
    pass_idx: Int = 0,
):
    def gather_one(i: Int) {imm}:
        _sppm_gather_one(vps, i, photons, heads, inv_cell, sd, sd.mediums, Int(sd.mediumCount),
                         sd.grids, sd.nvdbGrids,
                         sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
                         sd.spectral.cie_z, sd.spectral.d65, pass_wl, pass_idx)

    parallelize(gather_one, n_pix)


# ── Direct (NEE) lighting update ──────────────────────────────────────────────
# Mirrors pbrt-v4's SPPM "pixel.Ld" term: a shadow-ray light sample taken at
# each visible point EVERY SPPM pass (same cadence as the photon pass),
# summed here and divided by n_passes at finalize time. Needed because the
# photon-density (tau) term alone has to represent BOTH direct and indirect
# lighting through one noisy channel — pbrt keeps them separate, which is
# why its SPPM output is much smoother/less "blotchy" for the same pass
# count. Handles surface VPs (BRDF) and volume VPs (isotropic phase,
# albedo/4pi with no cosine) alike, each attenuated by the transmittance of
# whatever medium the VP sits in.


def _sppm_nee_update(
    vps:     Pointer[SPPMPixel, MutUntrackedOrigin],
    n_vps:   Int,
    ref sd:      SceneView,
    seed:    UInt64,
    pass_idx: Int,
):
    var n_lights = Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount)
    if n_lights == 0 and not (sd.sphereLightCount > 0):
        return

    def nee_one(i: Int) {imm}:
        var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + i), UInt64(11))
        _sppm_nee_one(vps, i, sd, pcg)

    parallelize(nee_one, n_vps)


def _sppm_render_core(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    verbose:  Bool,
) -> Tuple[Bool, Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]]:
    """Stochastic Progressive Photon Mapping main loop, factored out of
    `sppm_render` so the CPU driver's aux-buffer/denoise/write tail is a
    separate, reusable step. Returns (ok, global_pixels, caustic_pixels,
    albedo_pixels) -- `ok=False` (all pointers null/unusable) when the
    scene has no lights, mirroring the old function's `Int32(-1)` early
    return. `global_pixels`/`caustic_pixels` are the split
    _sppm_finalize_one_pixel returns (see its docstring); NEITHER carries
    the max-component sensor clamp yet -- the caller applies it ONCE, to
    their sum, after denoising `global_pixels` and adding `caustic_pixels`
    back unsmoothed. Same buffer contract as `_bdpt_render_core`
    (bdpt_*.mojo): caller-owned `n_pix*3` Float32 arrays, iso-scaled, NOT yet
    denoised."""
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)

    if Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount) == 0 and not (sd.sphereLightCount > 0):
        print("SPPM: no lights in scene, cannot emit photons")
        return Tuple[Bool, Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](
            False, Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Float32, MutUntrackedOrigin].unsafe_dangling())

    print("SPPM: " + String(fw) + "x" + String(fh)
          + " " + String(n_passes) + " passes x "
          + String(n_photons_per_pass) + " photons  r=" + String(initial_radius))

    # Allocate visible points (n_pix * _VP_SAMPLES independent samples) and photon buffer.
    #
    # max_photons is sized for the WORST CASE, not n_photons_per_pass: one
    # emitted photon path can store up to min(maxdepth, _MAX_B) - 1 deposits
    # (one per RR diffuse-diffuse bounce past the first), not just one. A
    # buffer sized at n_photons_per_pass (the older code here) meant the
    # shared atomic `counter` in _sppm_store_photon started silently
    # dropping deposits the moment total stored events -- not paths --
    # exceeded n_photons_per_pass, which for any scene with real multi-
    # bounce diffuse interreflection is most of them. Root-caused via a
    # closed-cavity radiative-equilibrium test (reflectance 1.0 everywhere,
    # one wall emitting L=1: every surface must read exactly 1.0, Russian
    # roulette never fires since albedo=1, so this scene hits the worst
    # case on every path) -- SPPM read 0.39 of that 1.0 with the old sizing
    # and 0.87 with this fix, path tracer (unaffected by this buffer) reads
    # 1.00 throughout. See project_photon_estimator_energy_gap memory.
    #
    # An earlier version of this comment claimed storing every bounce
    # event "measurably over-brightened the render" and kept the drop
    # deliberately -- that test predates several other estimator fixes
    # landed this session and was never re-verified against a ground
    # truth; the closed-cavity test above is the ground truth, and it
    # says the drop was strictly wrong, not a deliberate bias/cost
    # tradeoff. If a future regression looks like "over-brightening",
    # re-derive against an equilibrium or analytic scene before
    # reintroducing a cap -- don't just trust the old claim.
    var n_vps    = n_pix * _VP_SAMPLES
    var vps     = unsafe_alloc[SPPMPixel](n_vps)
    var max_bounces_per_photon = min(Int(psc[unsafe_offset=0].max_depth), _photon_depth_cap(sd))
    # A subsurface interior blows this budget wide open: its random-walk steps
    # are deliberately NOT charged to maxdepth (see _sppm_trace_photon's loop
    # header), so one photon entering skin deposits at every scatter for as
    # long as the walk survives -- hundreds of events, not `maxdepth` of them.
    # Sized for maxdepth alone, the shared atomic counter in
    # _sppm_store_photon starts dropping deposits almost immediately, and the
    # few paths that DID fit leave a wildly inflated local photon density that
    # the estimator still divides by the full emitted count: head.pbrt read
    # ~33x the pbrt reference, uniformly, at every pass count. Exactly the
    # failure the comment above describes, with a different cause.
    var has_sss_medium = False
    for mi in range(Int(sd.mediumCount)):
        if sd.mediums[unsafe_offset=mi].is_sss != Int32(0):
            has_sss_medium = True
            break
    if has_sss_medium:
        max_bounces_per_photon += SSS_WALK_ROUNDS
    var max_photons = n_photons_per_pass * max(max_bounces_per_photon, 1)
    var photons = unsafe_alloc[SPPMPhoton](max_photons)
    var heads   = unsafe_alloc[Int32](2 * _HSIZE)   # heads | counts, see grid_keep
    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / initial_radius

    # Trace the camera/visible-point samples ONCE for the whole render — see
    # _sppm_camera_pass's docstring for why a per-pass re-trace (the old
    # design) breaks SPPM's convergence guarantee.
    var cam_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
    _sppm_camera_pass(
        vps, n_pix, _VP_SAMPLES, psc[unsafe_offset=0].film_w,
        psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
        sd, init_r2, cam_seed, Int(psc[unsafe_offset=0].max_depth),
        film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                       psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
    )
    if verbose:
        var n_valid = 0
        for i in range(n_vps):
            if vps[unsafe_offset=i].valid != Int32(0): n_valid += 1
        print("SPPM: " + String(n_valid) + "/" + String(n_vps) + " visible points found")

    # Photon passes
    var prog = Progress(n_passes, "passes", quiet=verbose)
    for pass_idx in range(n_passes):
        var pass_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
        var n_stored = _sppm_photon_pass(photons, n_photons_per_pass, max_photons, sd, pass_seed, pass_idx, Int(psc[unsafe_offset=0].max_depth))
        if n_stored > 0:
            _build_grid(photons, n_stored, heads, inv_cell, pass_idx)
            _gather_update(vps, n_vps, photons, heads, inv_cell, sd, pass_wavelengths(pass_idx), pass_idx)
        var nee_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 0xBF58476D1CE4E5B9 + 3)
        _sppm_nee_update(vps, n_vps, sd, nee_seed, pass_idx)
        def flush_one(i: Int) {imm}:
            _sppm_flush_one(vps, i, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
                            sd.spectral.cie_z, sd.spectral.d65)
        parallelize(flush_one, n_vps)
        if verbose:
            print("SPPM: pass " + String(pass_idx + 1) + "/" + String(n_passes)
                  + " stored=" + String(n_stored))
        prog.update(pass_idx + 1)
    _ = prog.finish()

    # Assemble output image: average the _VP_SAMPLES independently-converged
    # samples per pixel (each sample's own r2/tau/N_acc converges correctly
    # since its position/surface is fixed for the whole render — see
    # _sppm_camera_pass's docstring). A sample that never found a diffuse/
    # volume hit (N_acc == 0, e.g. it reflected off the water into the void)
    # contributes 0 for that sample, same as a path tracer sample that misses
    # everything — the average over all samples is what correctly reproduces
    # the fresnel-weighted reflect/refract blend a real specular interface
    # would show.
    var out_global = unsafe_alloc[Float32](n_pix * 3)
    var out_caustic = unsafe_alloc[Float32](n_pix * 3)
    var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)

    def finalize_one(i: Int) {imm}:
        var (acc_g, acc_c) = _sppm_finalize_one_pixel(vps, i, _VP_SAMPLES, Int32(n_passes), iso_scale,
                                         sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                                         sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
        out_global[unsafe_offset=i * 3 + 0] = acc_g.r
        out_global[unsafe_offset=i * 3 + 1] = acc_g.g
        out_global[unsafe_offset=i * 3 + 2] = acc_g.b
        out_caustic[unsafe_offset=i * 3 + 0] = acc_c.r
        out_caustic[unsafe_offset=i * 3 + 1] = acc_c.g
        out_caustic[unsafe_offset=i * 3 + 2] = acc_c.b
        var alb = _sppm_finalize_albedo_one_pixel(vps, i, _VP_SAMPLES)
        albedo_pixels[unsafe_offset=i * 3 + 0] = alb.r
        albedo_pixels[unsafe_offset=i * 3 + 1] = alb.g
        albedo_pixels[unsafe_offset=i * 3 + 2] = alb.b

    parallelize(finalize_one, n_pix)

    heads.unsafe_free()
    photons.unsafe_free()
    vps.unsafe_free()
    return Tuple[Bool, Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](
        True, out_global, out_caustic, albedo_pixels)


def sppm_render(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """CLI-facing SPPM entry point: run the progressive photon-mapping
    estimator (`_sppm_render_core`), then denoise (SPPMPixel.alb as the
    albedo AOV, normals/depth from the same integrator-agnostic
    render_aux_buffers the plain path tracer and VCM use) and write."""
    var (ok, out_global, out_caustic, albedo_pixels) = _sppm_render_core(
        psc, sd, n_passes, n_photons_per_pass, initial_radius, verbose)
    if not ok:
        return Int32(-1)

    _ = finish_render(psc, sd, out_global, albedo_pixels, no_denoise, out_caustic)

    out_global.unsafe_free(); out_caustic.unsafe_free(); albedo_pixels.unsafe_free()
    return Int32(0)
