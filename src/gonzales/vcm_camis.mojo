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
from std.math import log, min
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
