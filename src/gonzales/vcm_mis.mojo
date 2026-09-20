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


# ── The two CAMERA-side weights for an ENVIRONMENT light ────────────────────
# Everything above carries MIS state along a subpath. These two spend it, at
# the only two places a camera path can see an infinite light: it samples one
# (NEE) or it escapes into one.
#
# Both were a bare `power_heuristic(pdf_a, pdf_b)` -- a TWO-strategy weight,
# beta=2, that knows only about NEE and BSDF sampling. Every other strategy
# here is combined with the BALANCE heuristic (bdpt.mojo's opening line: "MIS:
# balance heuristic over all valid connection strategies"), and the two
# families cannot partition unity: NEE and the escape divided a full 1.0
# between themselves, leaving no share for vertex merging or t=1 light
# tracing, which then added theirs on top.
#
# Derived and verified in Scenes/vcm_env_mis_derivation.py (exact to 1e-16
# against a direct enumeration of every strategy's path density). At one
# surface vertex there:
#
#     strategy          correct   power heuristic gave
#     escape (s=0)       0.5400   0.7995
#     NEE    (s=1)       0.2107   0.2005
#     t=1                0.0442
#     merging            0.2050
#
# The ESCAPE is the dominant error: the 0.26 it takes too much is almost
# exactly merging + t=1's combined share, 0.249.
#
# Both take `cos_at_light = 1`: an environment light is NOT finite, so it has
# no surface to take a cosine at -- the same "usedCosLight" case the light
# subpath's own origin takes when it seeds these carries.


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
