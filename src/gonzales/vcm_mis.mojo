"""VCM's MIS bookkeeping: the dVCM/dVC/dVM recursion, in one place.

A VCM subpath carries three running quantities (Georgiev et al. 2012 /
SmallVCM) that let any later vertex compute the MIS weight over every
connection and merging strategy without walking back along the path. The
recursion is short but easy to get subtly wrong, and it was written out by
hand at sixteen sites in bdpt.mojo -- once per material per subpath -- which
is how a material ends up with a rule the others do not have.

Three rules cover every vertex:

  arrival  (`vcm_arrival_carries`)  divide by the geometry cos at the vertex
  scatter  (`vcm_scatter_carries`)  the recursion proper, per sampled direction
  hop      (`bssrdf_hop_carries`)   a subsurface hop, which has no connection
                                    across it

Volume vertices and delta dielectrics deliberately sit OUTSIDE this
recursion (they reset dVCM to 0 rather than carrying a real weight) -- see
project_vcm_stage2_mis_derivation; that gap is documented, not an oversight
of this module.
"""
from .geometry import PI


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
def vcm_scatter_carries(dvcm: Float32, dvc: Float32, dvm: Float32,
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
    participate in have no density to weight them by."""
    if pdf_fwd_w <= Float32(1e-8):
        return (Float32(0.0), Float32(0.0), Float32(0.0))
    return (Float32(1.0) / pdf_fwd_w,
            cos_over_pdf * (dvc * pdf_rev_w + dvcm + mis_vm_weight_factor),
            cos_over_pdf * (dvm * pdf_rev_w + dvcm * mis_vc_weight_factor + Float32(1.0)))


@always_inline
def bssrdf_hop_carries(dvcm: Float32, dvc: Float32, dvm: Float32,
                       p_area: Float32, pdf_rev_entry: Float32,
                       mis_vc_weight_factor: Float32) -> Tuple[Float32, Float32, Float32]:
    """(dVCM, dVC, dVM) after a hop from an entry vertex to its exit point.

    dVC's rule is machine-checked against brute-force path pdfs to 4e-16 by
    Scenes/vcm_bssrdf_mis_derivation.py:

        dVC  = (1/p_A) * (dVC * pdf_rev_entry + dVCM)   area measure: no cos, no d^2
        dVCM = 0                                         no connection ACROSS the hop

    pdf_rev_entry is the entry vertex's own direction pdf toward its
    predecessor (the exit lobe's cos/pi). dVCM = 0 is the load-bearing part:
    keeping 1/p_A counts a connection across the hop that does not exist and
    biases every real strategy dark.

    dVM follows the same shape with the vertex-merging term at the ENTRY
    vertex dropped, because the entry is never a merge site. That half is NOT
    covered by the harness (which models connections only; merging is off by
    default), so treat it as derived, not verified."""
    var inv = Float32(1.0) / p_area
    return (Float32(0.0),
            inv * (dvc * pdf_rev_entry + dvcm),
            inv * (dvm * pdf_rev_entry + dvcm * mis_vc_weight_factor))


@always_inline
def bssrdf_exit_scatter_carries(dvcm: Float32, dvc: Float32, dvm: Float32,
                                p_area: Float32, cos_out: Float32,
                                mis_vc_weight_factor: Float32,
                                mis_vm_weight_factor: Float32) -> Tuple[Float32, Float32, Float32]:
    """(dVCM, dVC, dVM) after the exit vertex scatters a cosine-sampled ray.

    vcm_scatter_carries with one difference the harness establishes: the exit
    vertex's REVERSE density toward its predecessor is the hop's p_A (area
    measure, symmetric), not a direction pdf."""
    return vcm_scatter_carries(dvcm, dvc, dvm, PI, cos_out / PI, p_area,
                               mis_vc_weight_factor, mis_vm_weight_factor)
