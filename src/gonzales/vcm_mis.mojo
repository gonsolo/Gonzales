"""VCM's MIS bookkeeping: the dVCM/dVC/dVM recursion, in one place.

A VCM subpath carries three running quantities (Georgiev et al. 2012 /
SmallVCM) that let any later vertex compute the MIS weight over every
connection and merging strategy without walking back along the path. The
recursion is short but easy to get subtly wrong, and it was written out by
hand at sixteen sites in bdpt_*.mojo -- once per material per subpath -- which
is how a material ends up with a rule the others do not have.

Two rules cover every vertex:

  arrival  (`vcm_arrival_carries`)  divide by the geometry cos at the vertex
  scatter  (`vcm_scatter_carries`)  the recursion proper, per sampled direction

Volume vertices and delta dielectrics deliberately sit OUTSIDE this
recursion (they reset dVCM to 0 rather than carrying a real weight) -- see
project_vcm_stage2_mis_derivation; that gap is documented, not an oversight
of this module.
"""
from .geometry import PI
from .sampling import power_heuristic


@always_inline
def vcm_arrival_carries(dvcm: Float32, dvc: Float32, dvm: Float32,
                        cos_at_vertex: Float32) -> Tuple[Float32, Float32, Float32]:
    """(dVCM, dVC, dVM) on ARRIVING at a surface vertex.

    The carries reach the vertex in solid-angle measure; dividing by the
    geometry cos converts them to the area measure every later strategy
    compares in. A grazing cos is left alone rather than dividing by ~0: the
    weights it would produce are meaningless and the vertex contributes
    nothing anyway."""
    if cos_at_vertex <= Float32(1e-6):
        return (dvcm, dvc, dvm)
    return (dvcm / cos_at_vertex, dvc / cos_at_vertex, dvm / cos_at_vertex)


@always_inline
def vcm_scatter_carries[dvc0: Bool = False](
                        dvcm: Float32, dvc: Float32, dvm: Float32,
                        cos_over_pdf: Float32, pdf_fwd_w: Float32, pdf_rev_w: Float32,
                        mis_vc_weight_factor: Float32,
                        mis_vm_weight_factor: Float32) -> Tuple[Float32, Float32, Float32]:
    """(dVCM, dVC, dVM) after scattering into a sampled direction.

        dVC  = (cos_out/pdf_fwd) * (dVC * pdf_rev + dVCM + w_vm)
        dVM  = (cos_out/pdf_fwd) * (dVM * pdf_rev + dVCM * w_vc + 1)
        dVCM = 1 / pdf_fwd

    `cos_over_pdf` is passed in rather than divided here because it
    simplifies to a constant for the lobes that have one (pi for a cosine
    lobe, whose pdf is proportional to cos_out) and the caller already has
    that constant; computing it here would round differently.

    pdf_fwd_w/pdf_rev_w are the sampled direction's density and the density
    of sampling BACK toward the predecessor, both solid-angle. A degenerate
    forward density zeroes the carries: the strategies this vertex could
    participate in have no density to weight them by.

    `dvc0=True` (vcm_grid.mojo's _VCM_CAMIS build) repurposes the third slot as
    dVC0, the dVC recursion run with w_vm = 0:

        dVC0 = (cos_out/pdf_fwd) * (dVC0 * pdf_rev + dVCM)

    dVM itself is never read (every merge weight forms dVC / eta at the merge
    vertex instead, see _bdpt_merge_mis_weight), so its slot is free. dVC0 is
    the pairing-independent half of CAMIS's split dVC(eta) = dVC0 + sum_i
    eta_i mu_i (Scenes/vcm_camis_hybrid_derivation.py): dVC is AFFINE in the
    per-vertex eta, so the same recursion at eta = 0 IS the constant term --
    no second recursion and no closed form. The delta rule (scale by cos_out)
    and the arrival divide are identical for all three carries, so only this
    function needs the switch. Default False keeps the
    legacy dVM, byte for byte."""
    if pdf_fwd_w <= Float32(1e-8):
        return (Float32(0.0), Float32(0.0), Float32(0.0))
    comptime if dvc0:
        return (Float32(1.0) / pdf_fwd_w,
                cos_over_pdf * (dvc * pdf_rev_w + dvcm + mis_vm_weight_factor),
                cos_over_pdf * (dvm * pdf_rev_w + dvcm))
    return (Float32(1.0) / pdf_fwd_w,
            cos_over_pdf * (dvc * pdf_rev_w + dvcm + mis_vm_weight_factor),
            cos_over_pdf * (dvm * pdf_rev_w + dvcm * mis_vc_weight_factor + Float32(1.0)))


@always_inline
def vcm_env_nee_weight(pdf_bsdf_dir_w: Float32, pdf_bsdf_rev_w: Float32,
                       direct_pdf_w: Float32, emission_pdf_w: Float32,
                       cos_out: Float32, mis_vm_weight_factor: Float32,
                       dvcm: Float32, dvc: Float32) -> Float32:
    """Balance-heuristic weight for NEE from a camera vertex to an env light.

        w_light  = pdf_bsdf_dir / direct_pdf
        w_camera = (emission_pdf * cos_out / direct_pdf)
                   * (w_vm + dVCM + dVC * pdf_bsdf_rev)
        weight   = 1 / (w_light + 1 + w_camera)

    `w_light` is the BSDF-sampling strategy that could have found this same
    direction; `w_camera` is every strategy the camera subpath carries --
    crucially including `mis_vm_weight_factor`, merging's share, which the old
    two-strategy weight had no way to express.

    Takes the ARRIVAL carries at this vertex (d^2 and cos already applied),
    which is what the caller holds while shading it."""
    if direct_pdf_w <= Float32(1e-12):
        return Float32(0.0)
    var w_light = pdf_bsdf_dir_w / direct_pdf_w
    var w_camera = (emission_pdf_w * cos_out / direct_pdf_w) * (
        mis_vm_weight_factor + dvcm + dvc * pdf_bsdf_rev_w)
    return Float32(1.0) / (w_light + Float32(1.0) + w_camera)


@always_inline
def vcm_env_escape_weight(direct_pdf_w: Float32, emission_pdf_w: Float32,
                          dvcm_post_scatter: Float32,
                          dvc_post_scatter: Float32) -> Float32:
    """Balance-heuristic weight for a camera ray that ESCAPES into an env light.

        weight = 1 / (1 + direct_pdf * dVCM + emission_pdf * dVC)

    TRAP, and the single error that made the first attempt at this render 20%
    dark: these are the POST-SCATTER carries at the last real vertex -- one
    step LATER than the NEE weight at that same vertex -- and they take no
    d^2/cos, because nothing was arrived at and an environment light is at
    infinity. Passing the arrival carries instead is 20% low at one bounce and
    73% low at two (measured in the derivation harness)."""
    return Float32(1.0) / (Float32(1.0)
                           + direct_pdf_w * dvcm_post_scatter
                           + emission_pdf_w * dvc_post_scatter)


# ── ONE MIS policy for next-event estimation ────────────────────────────────
# Every _nee_weight_* helper in bxdf.mojo used to end the same way:
#
#     var mis_w = power_heuristic(ls.pdf, pdf_bsdf)
#     return f * Li * (cos * mis_w / ls.pdf)
#
# -- five near-identical copies, differing only in which lobe they evaluate.
# That is why giving VCM its correct balance-heuristic weight meant editing
# eight separate call sites, and why only one of them ever got edited: the
# weight was welded into each helper instead of being a parameter.
#
# MisPolicy makes it a parameter. A path tracer passes the default and gets
# byte-identical behaviour; VCM passes its carries and gets the weight that
# actually partitions unity across connection, merging and light tracing.


@fieldwise_init
struct MisPolicy(TrivialRegisterPassable):
    """How a NEE sample is weighted against the strategies competing with it.

    `is_vcm = False` is the two-strategy power heuristic (NEE vs BSDF
    sampling), which is correct for a path tracer: those ARE the only two
    strategies it runs. VCM runs four, so it needs the balance weight over all
    of them -- see vcm_env_nee_weight, derived exactly in
    Scenes/vcm_env_mis_derivation.py."""
    var is_vcm:         Bool
    var vm_weight:      Float32   # mis_vm_weight_factor: merging's share
    var dvcm:           Float32   # camera subpath carries, at THIS vertex
    var dvc:            Float32
    var emission_pdf_w: Float32   # the light's emission density
    var pdf_rev_w:      Float32   # this lobe's reverse density toward wo
    var sole:           Bool      # NEE is the ONLY strategy -- weight is 1
    # |cos| between the light sample and the GEOMETRIC normal, or 0 when the
    # vertex has none (hair). vcm_env_nee_weight's cos_out converts the
    # light's emission density to an area density at the surface -- a
    # density, so it wants the geometric normal, while the BSDF cosine the
    # NEE helpers compute is the SHADING one. Mixing them made NEE disagree
    # with merging and light tracing wherever a bump map tilts ns.
    var cos_geo:        Float32


@always_inline
def mis_policy_power() -> MisPolicy:
    """The path tracer's policy: NEE vs BSDF sampling, nothing else exists."""
    return MisPolicy(False, Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), False, Float32(0))


@always_inline
def mis_policy_sole() -> MisPolicy:
    """SPPM's policy at a visible point: NEE is the ONLY strategy there is.

    An MIS weight exists to stop two strategies double-counting the same
    path. Applying one where no second strategy exists does not redistribute
    the discarded share -- it deletes it. That is not a tuning error, it is
    an unbiasedness error, and it is worth stating loudly because the value
    it deletes looks plausible: an SPPM visible point TERMINATES the camera
    path (_sppm_trace_visible_point stores and breaks), and direct photons
    are gated out of the map (`n_events > 1`), so direct light at a VP has
    exactly one estimator and must carry full weight.

    Measured before this existed: every env-lit furnace read 0.1765 of its
    analytic answer. That number is derivable -- for a uniform environment
    (pdf_l = 1/4pi) against a diffuse lobe (pdf_b = mu/pi) the power
    heuristic collapses to 1/(1 + 16 mu^2), and

        2 * integral_0^1 mu / (1 + 16 mu^2) dmu  =  ln(17)/16  =  0.17708

    which is the measured 0.1765 to four figures. _sppm_nee_weight's VOLUME
    branch had always documented this and passed no weight; the SURFACE
    branch silently inherited _nee_weight_simple_spectral's power-heuristic
    default, so it applied to every sphere and infinite light SPPM ever
    shaded."""
    return MisPolicy(False, Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), True, Float32(0))


@always_inline
def nee_mis_weight(p: MisPolicy, ls_pdf: Float32, pdf_bsdf_w: Float32,
                   cos_out: Float32) -> Float32:
    """The MIS weight for one NEE sample, under whichever policy applies."""
    if p.sole:
        return Float32(1.0)
    if not p.is_vcm:
        return power_heuristic(ls_pdf, pdf_bsdf_w)
    var cos_dens = p.cos_geo if p.cos_geo > Float32(0) else cos_out
    return vcm_env_nee_weight(pdf_bsdf_w, p.pdf_rev_w, ls_pdf,
                              p.emission_pdf_w, cos_dens, p.vm_weight,
                              p.dvcm, p.dvc)
