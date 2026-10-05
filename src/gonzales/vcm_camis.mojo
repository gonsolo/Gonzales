"""CAMIS for VCM, trace-time half: the per-vertex records each subpath writes.

Plan deep-hugging-locket (stage S1), ground truth
Scenes/vcm_camis_hybrid_derivation.py (S0, 41/41 checks). CAMIS (Grittmann
et al. 2021) replaces every vertex's merging density eta_i by c_i(x) eta_i,
and c_i depends on BOTH subpaths -- which neither knows while it is traced.
It still fits in trace-time state because dVC is AFFINE in the eta_i:

    dVC_T(eta) = dVC0_T + sum_{i<T} eta_i mu_{i,T},   mu_{i,T} = B_i prod_{l=i+1}^{T-1} A_l
    B_l = cos_out_l / (pdf_fwd_l cos_{l+1}),          A_l = B_l pdf_rev_l

dVC0 (the same recursion at eta = 0) is carried in the old dVM slot
(vcm_mis.vcm_scatter_carries[dvc0=True]), and each scattered vertex keeps a
record of A_l and eta_l B_l -- S0's "Horner form" (section 4): the evaluator
rebuilds dVC' = dVC0 + S with S = S * A + c * etaB over the records, which is
twice as precise as the log form in Float32, cannot drift, and needs no
special case at a pdf_rev = 0 cut (A = 0 there zeroes every earlier term by
itself). Everything c needs besides that (log P(y) prefixes, reverse-edge
sums, keep probabilities) is recorded here too; S2's evaluator and S3's
weight sites read it. This file only BUILDS records; nothing here computes a
weight.

THE CLASS. CAMIS applies only to a full path whose light is an area emitter
and whose every vertex is a plain, non-delta, MIS-scoped surface vertex, with
no medium segment or null-interface crossing on any edge (the harness models
neither). Every other path keeps today's exact legacy weights (c = 1), which
is still a valid Eq. 11 estimator. Each subpath carries a sticky class bit;
records written after it drops are never read, so they are not maintained.

Conventions, all frozen by S0 ("CONVENTIONS FROZEN HERE"):
  r      = |y1 - lens| tan(1 deg), y1 the camera's first stored hit
  log_k  = log(pi r^2); P(edge) = min(pi r^2 p_A, 1), so log P = min(log_k + log p_A, 0)
  log P(y) of the camera prefix starts at 0 at y1 (the lens edge has P = 1)
  lb     = 1 / (N keep)   (N = light paths per pass, keep = thinning's keep)

ONE CONVENTION S0 GOT DIFFERENTLY FROM THE AUTHORS' CODE (for S2 to adopt).
The official implementation (~/src/MisForCorrelatedBidir, PdfRatioVcm.cs,
on SeeSharp 1.4.2's BidirPathPdfs) never clamps P(z_0) on its own: the light
subpath's first surface vertex's PdfFromAncestor is the JOINT emitter-ray
density (position x direction, in area measure at x1), and PdfRatio clamps
it once, min((pi r^2)^2 P_A p_A(z0 -> x1), 1) -- the `next *= acceptArea`
at i == numSurfaceVertices - 1. S0 clamps the two factors separately. The
records below keep log P_A (origin slot) and the first edge's density
(-log dVCM at lam = 1) apart, so either form can be evaluated; the fused
one is the reference's, and the emission-hit / s = 1 sites then need only
the product (S0's finding 5 becomes moot). The reference's primary (lens)
edge is clamped too, but its camera pdf is per PIXEL, so pi r^2 p_A ~ 1e2
there and it clamps to 1 -- S0's P(lens edge) = 1 in all but tiny images.
"""
from std.math import log, min, max, exp, expm1, inf
from std.collections import Array
from .geometry import PI

# tan(1 degree), Grittmann et al.'s Eq. 17 radius angle.
comptime CAMIS_TAN_1DEG = Float32(0.017455064928217585)


@always_inline
def camis_clamp_log_p(log_k: Float32, log_p_area: Float32) -> Float32:
    """log P(edge) = log min(pi r^2 p_A, 1) (Eq. 16), from log p_A."""
    return min(log_k + log_p_area, Float32(0))


# ── Camera subpath ──────────────────────────────────────────────────────────

@fieldwise_init
struct CamisCamRecord(TrivialRegisterPassable):
    """What camera vertex tau leaves behind for evaluations at later vertices
    (S0's CamScatter). Written when the camera scatters at tau, finalised on
    arriving at tau+1, whose cosine A and eta B need. Never read by an
    evaluation AT tau itself."""
    var a: Float32        # A_tau = (cos_out/pdf_fwd) pdf_rev / cos_{tau+1}
    var eta_b: Float32    # eta_tau B_tau = eta (cos_out/pdf_fwd) / cos_{tau+1}
    var log_keep: Float32 # log keep_tau: lb_tau = 1 / (N keep_tau)
    var log_py: Float32   # log P(y) for merging AT tau: prefix sum, 0 at y1
    var rc: Float32       # sum_{s=1..tau} log P(reverse edge s -> s-1), reset at a cut


@fieldwise_init
struct CamisCamCarry(TrivialRegisterPassable):
    """The camera subpath's running CAMIS registers (S0's CamArrival minus
    what the vertex itself already stores: dVCM, dVC, and dVC0 in .dVM).
    Threaded through _bdpt_camera_path_bounce as a `mut` parameter, exactly
    like dvcm_carry/dvc_carry."""
    var in_class: Bool
    var cut: Int32         # records tau >= cut are live (a pdf_rev = 0 at `cut` kills earlier ones)
    var log_k: Float32     # log(pi r^2), fixed at y1
    var log_py: Float32    # log P(y) at the vertex last arrived at
    var rc: Float32        # rc of the vertex last scattered
    var log_g_prev: Float32  # log(|cos at T-1| / d^2) of the edge that arrived at T
    var cos_out_prev: Float32  # cos_out of the last scatter, for the next arrival's log_g_prev


@always_inline
def camis_cam_carry_init() -> CamisCamCarry:
    """A camera path starts in the Class; everything else is set at y1."""
    return CamisCamCarry(True, Int32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))


@always_inline
def camis_cam_arrive[N: Int](
    mut c: CamisCamCarry, mut recs: Array[CamisCamRecord, N], tau: Int,
    dvcm_arrival: Float32, cos_v: Float32, t_hit: Float32, vertex_ok: Bool,
):
    """Arriving at STORED camera vertex tau (0 = y1), after the carries were
    divided by cos_v. `dvcm_arrival` is that divided dVCM: 1/p_A of the edge
    in, for tau > 0. `vertex_ok`: this vertex itself qualifies for the Class
    (MIS-scoped surface, non-delta, not a BSSRDF exit).

    At y1 the path has had no event before (else it would already be out of
    the Class), so t_hit IS |y1 - lens|."""
    if not vertex_ok or cos_v <= Float32(1e-6) or t_hit <= Float32(0) or tau >= N:
        c.in_class = False
    if not c.in_class:
        return
    if tau == 0:
        var r = t_hit * CAMIS_TAN_1DEG
        c.log_k = log(PI * r * r)
        c.log_py = Float32(0)
        c.rc = Float32(0)
        c.cut = Int32(0)
        return
    c.log_py = c.log_py + min(c.log_k - log(dvcm_arrival), Float32(0))
    c.log_g_prev = log(c.cos_out_prev / (t_hit * t_hit))
    # Finalise the previous vertex's record with THIS vertex's cosine.
    var inv_cos = Float32(1) / cos_v
    recs[tau - 1].a = recs[tau - 1].a * inv_cos
    recs[tau - 1].eta_b = recs[tau - 1].eta_b * inv_cos


@always_inline
def camis_cam_scatter[N: Int](
    mut c: CamisCamCarry, mut recs: Array[CamisCamRecord, N], tau: Int,
    eta: Float32, log_keep: Float32,
    cos_over_pdf: Float32, cos_out: Float32, pdf_fwd_w: Float32, pdf_rev_w: Float32,
):
    """Camera vertex tau scattered a NON-DELTA sample. `eta` is merging's MIS
    density here (the eta_x vcm_scatter_carries was given as w_vm), and
    `cos_over_pdf` the same value that call used, so the record rounds like
    the carry it decomposes. A and eta B are stored WITHOUT the next vertex's
    cosine; camis_cam_arrive divides it in."""
    if not c.in_class:
        return
    if pdf_fwd_w <= Float32(1e-8) or tau >= N:
        # vcm_scatter_carries zeroes every carry here, which is not the affine
        # recursion the records decompose -- leave the Class instead.
        c.in_class = False
        return
    if pdf_rev_w <= Float32(0):
        c.cut = Int32(tau)
    if tau == 0 or pdf_rev_w <= Float32(0):
        c.rc = Float32(0)   # no reverse edge to the lens; reset at a cut
    else:
        c.rc = c.rc + camis_clamp_log_p(c.log_k, log(pdf_rev_w) + c.log_g_prev)
    recs[tau] = CamisCamRecord(cos_over_pdf * pdf_rev_w, eta * cos_over_pdf, log_keep, c.log_py, c.rc)
    c.cos_out_prev = cos_out


# ── Light subpath ───────────────────────────────────────────────────────────

@fieldwise_init
struct CamisLightRecord(TrivialRegisterPassable):
    """One stored light vertex's CAMIS record: `lvc_camis[k]` parallels
    `lvc[k]` (same capacity, same per-path slot layout), 28 bytes.

    Fields a / eta_b / log_pa_rev describe the vertex's own SCATTER and are
    written when the NEXT vertex arrives (a needs its cosine); a vertex whose
    path ended there never gets them, and never needs them: the evaluation
    at a junction lambda walks only records 1..lambda-1.

    Deliberately NOT a field (S0 finding 3): log p_A of the forward edge INTO
    the vertex, which is -log(lvc[k].dVCM) on arrival.

    At the ORIGIN slot of an area light (lvc[k].is_light == 1) log_pa_rev
    holds log P_A instead, the emitter's position density -- P(z_0) in S0."""
    var a: Float32          # A_lam (see CamisCamRecord.a)
    var eta_b: Float32      # eta_lam B_lam
    var log_keep: Float32   # log keep_lam
    var log_pa_rev: Float32 # log p_A of the camera-direction edge lam -> lam-1 = log pdf_rev + log_g_rev (unclamped: r is the camera's)
    var log_g_rev: Float32  # log(|cos at lam-1| / d^2): the junction's own reverse edge needs it apart from pdf_rev (S0 finding 2)
    var cut: Int32          # records lam >= cut are live (light side needs its own, S0 finding 1)
    var flags: Int32        # bit 0: the light subpath up to and including this vertex is in the Class


comptime CAMIS_IN_CLASS = Int32(1)


@fieldwise_init
struct CamisLightCarry(TrivialRegisterPassable):
    """The light subpath's running CAMIS registers, threaded through
    _bdpt_light_path_bounce like its dvcm_carry. The scatter half of a record
    waits here until the next arrival supplies its cosine."""
    var in_class: Bool
    var cut: Int32
    var cos_out_prev: Float32   # cos_out of the last scatter (the emission cos at the origin)
    var log_g_rev: Float32      # the current vertex's log_g_rev, for its scatter's log_pa_rev
    var has_pending: Bool       # a scatter record below waits for the next arrival
    var pend_a: Float32
    var pend_eta_b: Float32
    var pend_log_pa_rev: Float32


@always_inline
def camis_light_carry_off() -> CamisLightCarry:
    """A light path outside the Class from its origin on (non-area light)."""
    return CamisLightCarry(False, Int32(1), Float32(0), Float32(0), False,
                           Float32(0), Float32(0), Float32(0))


@always_inline
def camis_light_origin(
    recs: Pointer[CamisLightRecord, MutUntrackedOrigin], slot: Int,
    log_pa0: Float32, cos_emit: Float32,
) -> CamisLightCarry:
    """An area light's origin, stored at `slot`: S0's LightOrigin. The
    emission is no scatter record (records run lam = 1..), so only P(z_0)'s
    density and the class bit are written; `cos_emit` feeds the first
    arrival's reverse-edge factor. cut starts at 1 (the first surface
    vertex), as in the harness."""
    recs[unsafe_offset=slot] = CamisLightRecord(
        Float32(0), Float32(0), Float32(0), log_pa0, Float32(0), Int32(1), CAMIS_IN_CLASS)
    return CamisLightCarry(True, Int32(1), cos_emit, Float32(0), False,
                           Float32(0), Float32(0), Float32(0))


@always_inline
def camis_light_arrive(
    mut c: CamisLightCarry, recs: Pointer[CamisLightRecord, MutUntrackedOrigin], slot: Int,
    cos_v: Float32, t_hit: Float32, log_keep: Float32, vertex_ok: Bool,
):
    """A light vertex was just stored at `slot` (after its carries were
    divided by cos_v). Writes its record -- ALWAYS, even outside the Class:
    the buffer lives across passes, so a slot not rewritten would hand S3 a
    stale class bit -- and finalises the previous vertex's scatter record."""
    if not vertex_ok or cos_v <= Float32(1e-6) or t_hit <= Float32(0):
        c.in_class = False
    if not c.in_class:
        c.has_pending = False
        recs[unsafe_offset=slot] = CamisLightRecord(
            Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), c.cut, Int32(0))
        return
    if c.has_pending and slot >= 1:
        var inv_cos = Float32(1) / cos_v
        recs[unsafe_offset=slot - 1].a = c.pend_a * inv_cos
        recs[unsafe_offset=slot - 1].eta_b = c.pend_eta_b * inv_cos
        recs[unsafe_offset=slot - 1].log_pa_rev = c.pend_log_pa_rev
    c.has_pending = False
    c.log_g_rev = log(c.cos_out_prev / (t_hit * t_hit))
    recs[unsafe_offset=slot] = CamisLightRecord(
        Float32(0), Float32(0), log_keep, Float32(0), c.log_g_rev, c.cut, CAMIS_IN_CLASS)


@always_inline
def camis_light_scatter(
    mut c: CamisLightCarry, lam: Int, eta: Float32,
    cos_over_pdf: Float32, cos_out: Float32, pdf_fwd_w: Float32, pdf_rev_w: Float32,
):
    """Light vertex `lam` (its lvc slot index) scattered a NON-DELTA sample;
    see camis_cam_scatter for the arguments."""
    if not c.in_class:
        return
    if pdf_fwd_w <= Float32(1e-8):
        c.in_class = False
        return
    if pdf_rev_w <= Float32(0):
        c.cut = Int32(lam)
    c.has_pending = True
    c.pend_a = cos_over_pdf * pdf_rev_w
    c.pend_eta_b = eta * cos_over_pdf
    c.pend_log_pa_rev = log(pdf_rev_w) + c.log_g_rev
    c.cos_out_prev = cos_out


# ── evaluation (stage S2): c and the weight sites ───────────────────────────
#
# Pure functions, mirroring vcm_mis.mojo's style and Scenes/
# vcm_camis_hybrid_derivation.py's eval_merge/_connect/_splat/_emission_hit
# (Horner form only -- S0's recommendation, see camis_light_side). S3 wires
# these into bdpt_*.mojo's real weight sites (_bdpt_merge_mis_weight,
# _bdpt_connect_mis_weight, the t=1 splat, the emission-hit escape); this
# file only computes the weight given records + junction data, which is why
# every function below takes explicit scalars for the ordinary VCM state
# (dVCM, legacy dVC, eta, keep) that a call site already has -- CAMIS only
# ever ADDS a `dVC0`/records pair alongside them (bdpt_*.mojo's dVM slot and
# lvc_camis), never replaces them; the `camis_on=False` / out-of-Class path
# below is exactly today's existing weight, byte for byte.
#
# Pinned against Tests/unit/test_vcm_camis.mojo, which reproduces
# EXPECT_MERGE_W_CAMIS from vcm_camis_hybrid_derivation.py's section_pinned().


@always_inline
def camis_c(log_py: Float32, log_pz: Float32, log_lb: Float32) -> Float32:
    """Eq. 13 + 15's logistic form: c = P(y)/P(x) = 1 / (1 + P(z)(1/P(y) - 1)),
    log(1/P(y) - 1) = -log P(y) + log(-expm1(log P(y))). Never forms P(y) or
    P(z) directly (either can underflow to 0 well before c does); log_pz =
    -inf (a zero-density light suffix, e.g. an emission pdf of 0) or log_py
    = 0 (P(y) = 1, the primary hit, Eq. 17) both return c = 1 without
    reaching exp. `log_lb` is -log(n_t keep): Eq. 17's per-vertex clamp
    c >= 1 / (n_t keep)."""
    if log_pz == -inf[DType.float32]() or log_py >= Float32(0):
        return Float32(1)
    var z = log_pz - log_py + log(-expm1(log_py))
    var c = Float32(1) / (Float32(1) + exp(z))
    return min(max(c, exp(log_lb)), Float32(1))


@always_inline
def camis_camera_side[N: Int](
    cam: CamisCamCarry, recs: Array[CamisCamRecord, N], num_scat: Int,
    dvc0: Float32, log_keep_t: Float32, n_t: Float32,
    log_pz_t: Float32, log_rev_t: Float32, force_c1: Bool,
) -> Tuple[Float32, Float32]:
    """(dVC'_T, c_T). dVC'_T = dVC0_T + sum_tau c_tau eta_tau B_tau
    A_{tau+1..T-1}, accumulated in Horner form (S = S*A + c*etaB) over
    tau = 0..num_scat-1. `cam` is the running carry AT T (cut/log_py/rc/
    log_k/in_class already match S0's CamArrival at T); `log_pz_t` and
    `log_rev_t` are the pairing-dependent quantities the call site derives
    from the light side and the junction (see camis_eval_merge etc. below)."""
    var log_nt = log(n_t)
    var S = Float32(0)
    for tau in range(num_scat):
        var rec = recs[tau]
        var c: Float32
        if force_c1 or tau < Int(cam.cut):
            c = Float32(1)
        else:
            c = camis_c(rec.log_py, log_pz_t + log_rev_t + (cam.rc - rec.rc),
                       -log_nt - rec.log_keep)
        S = S * rec.a + c * rec.eta_b
    var c_t = Float32(1) if force_c1 else camis_c(cam.log_py, log_pz_t, -log_nt - log_keep_t)
    return (dvc0 + S, c_t)


@always_inline
def camis_light_side[N: Int](
    scat: Array[CamisLightRecord, N], num_scat: Int,
    log_pa_fwd: Array[Float32, N], log_pa_fwd_arrival: Float32,
    log_pa0: Float32, arr_cut: Int32,
    dvc0: Float32, log_keep_lambda: Float32, n_t: Float32,
    log_k: Float32, log_py_base: Float32, log_rev_l: Float32, force_c1: Bool,
) -> Tuple[Float32, Float32, Float32]:
    """(dVC'_Lambda, log P(z) through Lambda, c_Lambda). `scat`/`log_pa_fwd`
    are the Lambda-1 stored vertices lam = 1..Lambda-1 (index i = lam - 1);
    `log_pa_fwd[i]` is scat[i]'s own edge-in density (S0 finding 3: not
    stored in the record itself -- the call site derives it as
    -log(lvc[slot].dVCM)). `log_pa_fwd_arrival` is Lambda's own.
    `log_py_base`/`log_rev_l` are pairing-dependent, from the camera side and
    the junction. Two passes, like the harness: a backward one for log P(y)
    at every scat vertex (needs the suffix, i.e. later vertices first), then
    a forward one that grows log P(z) as a running scalar (each index is
    read exactly once) alongside the Horner accumulation."""
    var log_nt = log(n_t)
    var lpy = Array[Float32, N](fill=Float32(0))
    var acc = log_py_base + log_rev_l
    for i in range(num_scat - 1, -1, -1):
        lpy[i] = acc
        acc = acc + camis_clamp_log_p(log_k, scat[i].log_pa_rev)
    var lpz_run = camis_clamp_log_p(log_k, log_pa0)
    var S = Float32(0)
    for i in range(num_scat):
        var rec = scat[i]
        var lam = i + 1
        var c: Float32
        if force_c1 or lam < Int(arr_cut):
            c = Float32(1)
        else:
            c = camis_c(lpy[i], lpz_run, -log_nt - rec.log_keep)
        S = S * rec.a + c * rec.eta_b
        lpz_run = lpz_run + camis_clamp_log_p(log_k, log_pa_fwd[i])
    lpz_run = lpz_run + camis_clamp_log_p(log_k, log_pa_fwd_arrival)
    var c_l = Float32(1) if force_c1 else camis_c(log_py_base, lpz_run, -log_nt - log_keep_lambda)
    return (dvc0 + S, lpz_run, c_l)


@always_inline
def camis_eval_merge[NC: Int, NL: Int](
    cam: CamisCamCarry, cam_scat: Array[CamisCamRecord, NC], num_cam_scat: Int,
    cam_dvcm: Float32, cam_dvc_legacy: Float32, cam_dvc0: Float32,
    cam_eta: Float32, cam_log_keep: Float32,
    light_arr: CamisLightRecord, light_scat: Array[CamisLightRecord, NL], num_light_scat: Int,
    light_log_pa_fwd: Array[Float32, NL], light_log_pa_fwd_arrival: Float32,
    light_log_pa0: Float32, light_origin_in_class: Bool,
    light_dvcm: Float32, light_dvc_legacy: Float32, light_dvc0: Float32, light_eta: Float32,
    cam_dir_w: Float32, cam_rev_w: Float32,
    n_t: Float32, camis_on: Bool, force_c1: Bool = False,
) -> Float32:
    """_bdpt_merge_mis_weight's CAMIS branch: merge a photon (light side) at
    camera vertex T (`cam`). Mirrors eval_merge in
    Scenes/vcm_camis_hybrid_derivation.py; pinned by
    Tests/unit/test_vcm_camis.mojo against EXPECT_MERGE_W_CAMIS."""
    var use = (camis_on and cam.in_class
              and (light_arr.flags & CAMIS_IN_CLASS) != 0 and light_origin_in_class)
    var dvc_l: Float32
    var dvc_c: Float32
    var c_t: Float32
    if use:
        var log_rev_l = camis_clamp_log_p(cam.log_k, log(cam_dir_w) + light_arr.log_g_rev)
        var light_result = camis_light_side(
            light_scat, num_light_scat, light_log_pa_fwd, light_log_pa_fwd_arrival,
            light_log_pa0, light_arr.cut, light_dvc0, light_arr.log_keep, n_t,
            cam.log_k, cam.log_py, log_rev_l, force_c1)
        dvc_l = light_result[0]
        var lpz = light_result[1]
        var log_rev_t = camis_clamp_log_p(cam.log_k, log(cam_rev_w) + cam.log_g_prev)
        var cam_result = camis_camera_side(
            cam, cam_scat, num_cam_scat, cam_dvc0, cam_log_keep, n_t,
            lpz, log_rev_t, force_c1)
        dvc_c = cam_result[0]
        c_t = cam_result[1]
    else:
        dvc_l = light_dvc_legacy
        dvc_c = cam_dvc_legacy
        c_t = Float32(1)
    var eta = c_t * cam_eta                # c into eta_scale, never into inv_eta_x
    var w_light = (light_dvcm + dvc_l * cam_dir_w) / eta
    var w_camera = (cam_dvcm + dvc_c * cam_rev_w) / eta
    return Float32(1) / (w_light + Float32(1) + w_camera)


@always_inline
def camis_eval_connect[NC: Int, NL: Int](
    cam: CamisCamCarry, cam_scat: Array[CamisCamRecord, NC], num_cam_scat: Int,
    cam_dvcm: Float32, cam_dvc_legacy: Float32, cam_dvc0: Float32,
    cam_eta: Float32, cam_log_keep: Float32,
    light_arr: CamisLightRecord, light_scat: Array[CamisLightRecord, NL], num_light_scat: Int,
    light_log_pa_fwd: Array[Float32, NL], light_log_pa_fwd_arrival: Float32,
    light_log_pa0: Float32, light_origin_in_class: Bool,
    light_dvcm: Float32, light_dvc_legacy: Float32, light_dvc0: Float32, light_eta: Float32,
    cam_dir_a: Float32, cam_rev_w: Float32, light_dir_a: Float32, light_rev_w: Float32,
    n_t: Float32, camis_on: Bool, force_c1: Bool = False,
) -> Float32:
    """_connect between a stored light vertex (`light_arr`, s >= 2) and
    camera vertex T (`cam`). For s = 1 (the light vertex IS the emitter, no
    light-side records at all) use camis_eval_connect_s1 instead."""
    var use = (camis_on and cam.in_class
              and (light_arr.flags & CAMIS_IN_CLASS) != 0 and light_origin_in_class)
    var log_k = cam.log_k
    var w_light: Float32
    var lpz: Float32
    if use:
        var log_rev_l = camis_clamp_log_p(log_k, log(light_rev_w) + light_arr.log_g_rev)
        var log_py_base = cam.log_py + camis_clamp_log_p(log_k, log(cam_dir_a))
        var light_result = camis_light_side(
            light_scat, num_light_scat, light_log_pa_fwd, light_log_pa_fwd_arrival,
            light_log_pa0, light_arr.cut, light_dvc0, light_arr.log_keep, n_t,
            log_k, log_py_base, log_rev_l, force_c1)
        var dvc_l = light_result[0]
        lpz = light_result[1]
        var c_l = light_result[2]
        w_light = cam_dir_a * (c_l * light_eta + light_dvcm + dvc_l * light_rev_w)
    else:
        lpz = Float32(0)   # unused: dvc_c below takes the legacy branch too
        w_light = cam_dir_a * (light_eta + light_dvcm + light_dvc_legacy * light_rev_w)
    var dvc_c: Float32
    var c_t: Float32
    if use:
        var log_rev_t = camis_clamp_log_p(log_k, log(cam_rev_w) + cam.log_g_prev)
        var cam_result = camis_camera_side(
            cam, cam_scat, num_cam_scat, cam_dvc0, cam_log_keep, n_t,
            lpz + camis_clamp_log_p(log_k, log(light_dir_a)), log_rev_t, force_c1)
        dvc_c = cam_result[0]
        c_t = cam_result[1]
    else:
        dvc_c = cam_dvc_legacy
        c_t = Float32(1)
    var w_camera = light_dir_a * (c_t * cam_eta + cam_dvcm + dvc_c * cam_rev_w)
    return Float32(1) / (w_light + Float32(1) + w_camera)


@always_inline
def camis_eval_connect_s1[NC: Int](
    cam: CamisCamCarry, cam_scat: Array[CamisCamRecord, NC], num_cam_scat: Int,
    cam_dvcm: Float32, cam_dvc_legacy: Float32, cam_dvc0: Float32,
    cam_eta: Float32, cam_log_keep: Float32,
    origin_log_pa0: Float32, origin_direct_pdf_a: Float32, origin_in_class: Bool,
    cam_dir_a: Float32, cam_rev_w: Float32, light_dir_a: Float32,
    n_t: Float32, camis_on: Bool, force_c1: Bool = False,
) -> Float32:
    """_connect, s = 1: the light vertex IS the emitter (SmallVCM's
    DirectIllumination form) -- no light-side records exist to walk."""
    var use = camis_on and cam.in_class and origin_in_class
    var log_k = cam.log_k
    var w_light = cam_dir_a / origin_direct_pdf_a
    var lpz = camis_clamp_log_p(log_k, origin_log_pa0)
    var dvc_c: Float32
    var c_t: Float32
    if use:
        var log_rev_t = camis_clamp_log_p(log_k, log(cam_rev_w) + cam.log_g_prev)
        var cam_result = camis_camera_side(
            cam, cam_scat, num_cam_scat, cam_dvc0, cam_log_keep, n_t,
            lpz + camis_clamp_log_p(log_k, log(light_dir_a)), log_rev_t, force_c1)
        dvc_c = cam_result[0]
        c_t = cam_result[1]
    else:
        dvc_c = cam_dvc_legacy
        c_t = Float32(1)
    var w_camera = light_dir_a * (c_t * cam_eta + cam_dvcm + dvc_c * cam_rev_w)
    return Float32(1) / (w_light + Float32(1) + w_camera)


@always_inline
def camis_eval_splat[NL: Int](
    light_arr: CamisLightRecord, light_scat: Array[CamisLightRecord, NL], num_light_scat: Int,
    light_log_pa_fwd: Array[Float32, NL], light_log_pa_fwd_arrival: Float32,
    light_log_pa0: Float32, light_origin_in_class: Bool,
    light_dvcm: Float32, light_dvc_legacy: Float32, light_dvc0: Float32, light_eta: Float32,
    log_k: Float32, cam_pdf_a: Float32, rev_w: Float32, n_splat: Float32,
    n_t: Float32, camis_on: Bool, force_c1: Bool = False,
) -> Float32:
    """t = 1: the light vertex `light_arr` is seen directly by the lens. log
    P(y) of every light vertex starts from the primary edge's P = 1 (0),
    same as camis_eval_merge/_connect's camera side at y1."""
    var use = camis_on and (light_arr.flags & CAMIS_IN_CLASS) != 0 and light_origin_in_class
    var dvc_l: Float32
    var c_l: Float32
    if use:
        var log_rev = camis_clamp_log_p(log_k, log(rev_w) + light_arr.log_g_rev)
        var light_result = camis_light_side(
            light_scat, num_light_scat, light_log_pa_fwd, light_log_pa_fwd_arrival,
            light_log_pa0, light_arr.cut, light_dvc0, light_arr.log_keep, n_t,
            log_k, Float32(0), log_rev, force_c1)
        dvc_l = light_result[0]
        c_l = light_result[2]
    else:
        dvc_l = light_dvc_legacy
        c_l = Float32(1)
    var w_light = (cam_pdf_a / n_splat) * (c_l * light_eta + light_dvcm + dvc_l * rev_w)
    return Float32(1) / (w_light + Float32(1))


@always_inline
def camis_eval_emission_hit[NC: Int](
    cam: CamisCamCarry, cam_scat: Array[CamisCamRecord, NC], num_cam_scat: Int,
    cam_dvcm: Float32, cam_dvc_legacy: Float32, cam_dvc0: Float32, cam_log_keep: Float32,
    direct_pdf_a: Float32, emission_pdf_w: Float32, log_p_emit: Float32,
    n_t: Float32, camis_on: Bool, force_c1: Bool = False,
) -> Float32:
    """s = 0: the camera ray hit the emitter directly (SmallVCM's
    GetLightRadiance). The 'light subpath' merged against is z_0 alone, so
    there is no light-side call -- only the camera side, walked with the
    emitter's own position/directional densities as the pairing-dependent
    P(z) and reverse edge."""
    var use = camis_on and cam.in_class
    var dvc_c: Float32
    if use:
        var log_k = cam.log_k
        var log_pz = camis_clamp_log_p(log_k, log(direct_pdf_a))
        var log_rev = camis_clamp_log_p(log_k, log_p_emit + cam.log_g_prev)
        var cam_result = camis_camera_side(
            cam, cam_scat, num_cam_scat, cam_dvc0, cam_log_keep, n_t,
            log_pz, log_rev, force_c1)
        dvc_c = cam_result[0]
    else:
        dvc_c = cam_dvc_legacy
    return Float32(1) / (Float32(1) + direct_pdf_a * cam_dvcm + emission_pdf_w * dvc_c)
