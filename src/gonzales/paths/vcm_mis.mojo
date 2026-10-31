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

Delta dielectrics reset dVCM to 0. Volume vertices carry the same recursion with
no cosine (Scenes/vcm_volume_mis_derivation.py).
"""
from gonzales.core.geometry import PI, Vec3f
from gonzales.scene.appearance.media import hg_sample


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
def vcm_phase_scatter[dvc0: Bool = False](
                        wo: Vec3f, g: Float32, u1: Float32, u2: Float32,
                        dvcm: Float32, dvc: Float32, dvm: Float32,
                        mis_vc_weight_factor: Float32, eta: Float32) -> Tuple[Float32, Float32, Float32, Vec3f]:
    """A volume vertex's scatter: the carries and the new direction. No cosine, and the phase function is symmetric."""
    var hs = hg_sample(wo, g, u1, u2)
    var (a, b, c) = vcm_scatter_carries[dvc0=dvc0](dvcm, dvc, dvm, Float32(1) / hs[3], hs[3], hs[3], mis_vc_weight_factor, eta)
    return (a, b, c, Vec3f(hs[0], hs[1], hs[2]))


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
