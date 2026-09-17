"""Diffusion BSSRDF — subsurface transport evaluated AT RENDER TIME from a
surface-side estimate, instead of by photon-mapping the interior.

Why this exists. gonzales' SPPM photon-maps the INSIDE of a subsurface object:
photons random-walk through the skin depositing volume photons, and the camera
ray stops at its first collision to place a VOLUME visible point about one mean
free path below the surface. That makes the estimate a 3D density estimate in a
medium whose mean free path is ~0.001 scene units, so the gather sphere's volume
goes as r^3 and the photon count needed explodes. Measured on head.pbrt, the
result is insensitive to photon count -- 0.068x the reference at 36864
photons/pass and 0.068x at 589824 -- i.e. biased, not merely noisy.

The classical alternative (Jensen et al. 2001; Jensen & Buhler 2002, which is
what pbrt does via its photon-beam-diffusion table) inverts that: photons land
on the SURFACE, and subsurface transport is evaluated at render time by a
diffusion kernel. The photon map stays two-dimensional, where density
estimation works at ordinary photon counts, and nothing has to resolve a
0.001-unit mean free path.

Better still for a progressive estimator: `dipole_rd` below IS a normalised
kernel (it integrates over the plane to the diffuse albedo), so a BSSRDF gather
needs NO 1/(pi r^2) and NO shrinking radius. The radius is a truncation
distance chosen from the material's own diffusion length, not a bias parameter
-- which removes the bias/radius tradeoff that makes the volumetric path fail
here in the first place.

This is the CLASSICAL dipole (Jensen 2001). It is the simple member of the
family; pbrt uses photon beam diffusion (Habel et al. 2013), which is more
accurate near the source and for high absorption. Swapping the kernel later
means replacing `dipole_rd` alone -- every caller only wants R_d(r).
"""

from std.math import exp, sqrt, max, min, log, cos, sin
from gonzales.geometry import RGB, Vec3f, fr_dielectric, PI


@always_inline
def fdr_moment(eta: Float32) -> Float32:
    """Diffuse Fresnel reflectance F_dr(eta): the fraction of internally
    diffuse light the boundary reflects back in. Jensen et al. 2001 eq. 3's
    polynomial fit, valid for eta >= 1 (denser medium below the surface)."""
    if eta >= Float32(1.0):
        return (Float32(-1.4399) / (eta * eta) + Float32(0.7099) / eta
                + Float32(0.6681) + Float32(0.0636) * eta)
    # eta < 1 branch of the same fit, kept so a caller cannot get a silently
    # nonsensical A below.
    return (Float32(-0.4399) + Float32(0.7099) / eta - Float32(0.3319) / (eta * eta)
            + Float32(0.0636) / (eta * eta * eta))


@always_inline
def dipole_rd_channel(sigma_s: Float32, sigma_a: Float32, g: Float32,
                      eta: Float32, r: Float32) -> Float32:
    """Classical dipole diffuse reflectance R_d(r) for ONE channel: the
    fraction of power entering at a point that leaves a distance `r` away
    along the surface, per unit area.

    Uses the reduced (similarity-theory) coefficients, which is what makes a
    diffusion approximation legitimate for an anisotropic phase function:
    sigma_s' = sigma_s (1 - g). The two poles are a real source one mean free
    path below the surface and a virtual one above it, placed so the
    linearised boundary condition holds."""
    var ss_p = sigma_s * (Float32(1.0) - g)          # reduced scattering
    var st_p = ss_p + sigma_a                        # reduced extinction
    if st_p <= Float32(0.0) or ss_p <= Float32(0.0):
        return Float32(0.0)
    var alpha_p = ss_p / st_p                        # reduced albedo
    var sigma_tr = sqrt(Float32(3.0) * sigma_a * st_p)   # effective transport
    var fdr = fdr_moment(eta)
    var A = (Float32(1.0) + fdr) / (Float32(1.0) - fdr)
    var z_r = Float32(1.0) / st_p                    # real source depth
    var z_v = z_r * (Float32(1.0) + Float32(4.0) / Float32(3.0) * A)  # virtual
    var r2 = r * r
    var d_r = sqrt(r2 + z_r * z_r)
    var d_v = sqrt(r2 + z_v * z_v)
    if d_r <= Float32(1e-12) or d_v <= Float32(1e-12):
        return Float32(0.0)
    var term_r = z_r * (sigma_tr * d_r + Float32(1.0)) * exp(-sigma_tr * d_r) / (d_r * d_r * d_r)
    var term_v = z_v * (sigma_tr * d_v + Float32(1.0)) * exp(-sigma_tr * d_v) / (d_v * d_v * d_v)
    var rd = alpha_p / (Float32(4.0) * Float32(3.14159265) ) * (term_r + term_v)
    return max(rd, Float32(0.0))


@always_inline
def dipole_rd(sigma_s: RGB, sigma_a: RGB, g: Float32, eta: Float32, r: Float32) -> RGB:
    """Per-channel R_d(r). Channels are independent: the dipole is a scalar
    diffusion solution and skin's channels have very different mean free
    paths, which is exactly where the colour comes from."""
    return RGB(dipole_rd_channel(sigma_s.r, sigma_a.r, g, eta, r),
               dipole_rd_channel(sigma_s.g, sigma_a.g, g, eta, r),
               dipole_rd_channel(sigma_s.b, sigma_a.b, g, eta, r))


@always_inline
def dipole_max_radius(sigma_s: RGB, sigma_a: RGB, g: Float32) -> Float32:
    """A truncation distance for the gather: where R_d has decayed to
    insignificance. This is NOT a density-estimation radius -- R_d is already
    normalised, so this only bounds the search, it does not scale the result.
    Taken as a few effective transport mean free paths of the LONGEST-reaching
    channel (smallest sigma_tr), so no channel is clipped."""
    var best = Float32(0.0)
    for c in range(3):
        var ss = sigma_s.r if c == 0 else (sigma_s.g if c == 1 else sigma_s.b)
        var sa = sigma_a.r if c == 0 else (sigma_a.g if c == 1 else sigma_a.b)
        var ss_p = ss * (Float32(1.0) - g)
        var st_p = ss_p + sa
        if st_p <= Float32(0.0):
            continue
        var sigma_tr = sqrt(Float32(3.0) * sa * st_p)
        var reach = (Float32(1.0) / st_p) if sigma_tr <= Float32(1e-8) else (Float32(1.0) / sigma_tr)
        if reach > best:
            best = reach
    # Four transport lengths captures essentially all of the profile's energy
    # while keeping the neighbour search bounded.
    return Float32(4.0) * best


# ── Sampling the profile ────────────────────────────────────────────────────
# A gather only needs to EVALUATE R_d. A bidirectional integrator has to
# SAMPLE it: the camera subpath arrives at an entry point and must produce an
# exit point (and vice versa for the light subpath) with a known pdf, so the
# hop can carry a throughput weight and take part in MIS. That is the whole
# difference between the SPPM path (evaluate) and this one (sample).
#
# The profile is sampled radially as an exponential in the effective transport
# coefficient sigma_tr -- which is exactly R_d's asymptotic decay, so the
# weight R_d/pdf stays bounded -- and uniformly in azimuth. The pdf returned is
# an AREA density on the tangent plane (the 1/(2 pi r) Jacobian is already
# folded in), which is what a surface-area-measure integrator wants.

@always_inline
def dipole_sigma_tr_channel(sigma_s: Float32, sigma_a: Float32, g: Float32) -> Float32:
    var ss_p = sigma_s * (Float32(1.0) - g)
    var st_p = ss_p + sigma_a
    if st_p <= Float32(0.0):
        return Float32(0.0)
    return sqrt(Float32(3.0) * sigma_a * st_p)


@always_inline
def dipole_sample_radius(sigma_tr: Float32, u: Float32) -> Float32:
    """Radius from an exponential of rate sigma_tr: r = -ln(1-u)/sigma_tr."""
    if sigma_tr <= Float32(0.0):
        return Float32(0.0)
    var uu = min(max(u, Float32(0.0)), Float32(0.9999999))
    return -log(Float32(1.0) - uu) / sigma_tr


@always_inline
def dipole_radius_pdf_area(sigma_tr: Float32, r: Float32) -> Float32:
    """Area-measure pdf of `dipole_sample_radius` on the tangent plane:
    p_area(r) = sigma_tr * exp(-sigma_tr r) / (2 pi r).

    Diverges as r -> 0, which is correct (the sampler concentrates there) but
    has to be guarded by the caller, since R_d is finite at the origin and the
    ratio would otherwise be 0/0."""
    if sigma_tr <= Float32(0.0) or r <= Float32(1e-9):
        return Float32(0.0)
    return sigma_tr * exp(-sigma_tr * r) / (Float32(2.0) * Float32(3.14159265) * r)


@always_inline
def dipole_mis_sigma_tr(sigma_s: RGB, sigma_a: RGB, g: Float32, which: Int) -> Float32:
    """sigma_tr of one channel, for picking which channel's profile to sample
    from. Sampling a single channel and weighting by the balance heuristic over
    all three is what keeps a chromatic material (skin's channels differ by
    ~4x in transport length) from blowing up -- the same reasoning as the
    hero-wavelength free-flight MIS."""
    if which == 0: return dipole_sigma_tr_channel(sigma_s.r, sigma_a.r, g)
    if which == 1: return dipole_sigma_tr_channel(sigma_s.g, sigma_a.g, g)
    return dipole_sigma_tr_channel(sigma_s.b, sigma_a.b, g)


@always_inline
def dipole_sample_pdf_mis(sigma_s: RGB, sigma_a: RGB, g: Float32, r: Float32) -> Float32:
    """The MIXTURE area pdf actually used: a channel is chosen uniformly and
    its exponential sampled, so the density is the average of the three. Using
    one channel's pdf alone would leave the other two's weights unbounded."""
    var acc = Float32(0.0)
    var n = 0
    for c in range(3):
        var str_c = dipole_mis_sigma_tr(sigma_s, sigma_a, g, c)
        if str_c > Float32(0.0):
            acc += dipole_radius_pdf_area(str_c, r)
            n += 1
    if n == 0:
        return Float32(0.0)
    return acc / Float32(n)


# ── Exit-point sampling (the bidirectional half) ────────────────────────────
# MIS DERIVATION, because this is the part that cannot be waved through.
#
# VCM's per-vertex carries assume a vertex whose direction was sampled with a
# SOLID-ANGLE pdf, converted to area measure by the geometry term
# G = cos(theta) / d^2:
#
#     dVCM' = 1 / p_area          with p_area = p_omega * G
#     dVC'  = (cos_out / p_area) * (dVC * p_rev + dVCM + w_vm)
#
# A BSSRDF hop is not that. It is sampled NATIVELY in area measure -- we draw
# a radius and an azimuth on the tangent plane and land on the surface -- so
# its forward pdf p_A(x_o | x_i) is ALREADY an area density and the geometry
# conversion factor is exactly 1. There is no cos/d^2 to apply, and applying
# one (as a naive port of the local-vertex recursion would) is wrong by
# precisely that factor.
#
# The hop is therefore treated as an EXTRA vertex on the subpath, not as a
# modified local one:
#
#     x_i  (entry)  --p_A-->  x_o  (exit)  --cosine-->  next direction
#
# with, at the hop,
#
#     dVCM' = 1 / p_A
#     dVC'  = (1 / p_A) * (dVC * p_rev + dVCM + w_vm)        [no cos_out]
#     dVM'  = (1 / p_A) * (dVM * p_rev + dVCM * w_vc + 1)    [no cos_out]
#
# and p_rev = p_A. That needs checking rather than asserting, because the
# radial profile being symmetric is NOT by itself enough -- the probe Jacobian
# could break it. Write both out:
#
#     p_A(x_o | x_i) = p_radial(r) * |n_o . axis_i|,   axis_i = n_i
#     p_A(x_i | x_o) = p_radial(r) * |n_i . axis_o|,   axis_o = n_o
#
# Both cosines are |n_i . n_o|, and p_radial depends only on r = |x_i - x_o|,
# so the two are equal -- for CURVED surfaces too, not just planar ones. The
# hop is genuinely symmetric under normal-axis probing. That is what makes the
# bidirectional case tractable: a light subpath's hop has the same density as
# a camera subpath's, so a connection through a subsurface object needs no
# separate reverse profile. (It would NOT survive pbrt's three-axis probe MIS,
# where the axis choice differs at the two ends -- a reason to keep the single
# normal axis here beyond simplicity.)
#
# The exit vertex x_o is then an ORDINARY diffuse-like vertex (cosine exit
# lobe) and takes the standard recursion unchanged.
#
# The probe. Sampling a radius gives a point on the TANGENT PLANE, which is
# not on the surface. A probe ray along the normal axis finds the real exit
# point, and the change of variables from the tangent disk to the surface
# contributes |cos(theta_probe)| -- the angle between the surface normal at
# x_o and the probe axis. That factor is in `pdf_area` below. (pbrt additionally
# MIS-combines probes along three axes, which recovers the grazing geometry a
# single axis under-samples; single-axis is unbiased wherever the probe lands
# and loses the near-tangential cases, so this is a known, bounded gap.)

@always_inline
def bssrdf_probe_offset(r: Float32, phi: Float32, r_max: Float32,
                        t: Vec3f, b: Vec3f, n: Vec3f) -> Tuple[Vec3f, Float32]:
    """Probe segment for an exit point at radius `r`, azimuth `phi` around a
    surface frame. Returns the START offset from the entry point and the
    segment LENGTH: a chord of the sphere of radius `r_max`, centred on the
    entry point, so every surface point within the profile's reach can be
    found. Probing along the normal is what makes the tangent-disk sample a
    surface sample."""
    var half = sqrt(max(r_max * r_max - r * r, Float32(0.0)))
    var lateral = t * (r * cos(phi)) + b * (r * sin(phi))
    return (lateral + n * half, Float32(2.0) * half)


@always_inline
def bssrdf_exit_pdf_area(sigma_s: RGB, sigma_a: RGB, g: Float32,
                         r: Float32, cos_probe: Float32) -> Float32:
    """p_A(x_o | x_i): the area density of the sampled exit point ON THE
    SURFACE. The radial mixture density is a density on the tangent PLANE, so
    the Jacobian of the plane->surface map, |cos(theta_probe)|, converts it.
    Returns 0 for a degenerate (edge-on) probe, which the caller must treat as
    a failed sample rather than an infinite weight."""
    var ct = abs(cos_probe)
    if ct <= Float32(1e-4):
        return Float32(0.0)
    return dipole_sample_pdf_mis(sigma_s, sigma_a, g, r) * ct


# ── The exit lobe and the VCM carries across a hop ──────────────────────────
# Both consumers of these (VCM's camera and light subpaths) must agree bit for
# bit, because the MIS weights assume the two sides use the same densities.
# That is why they live here and not inline in either subpath.

@always_inline
def bssrdf_exit_ft(cos_theta: Float32, eta: Float32) -> Float32:
    """Fresnel transmittance through the boundary, the angular factor of both
    the entry and the exit lobe. The exit vertex's BSDF is Ft(cos)/pi: its
    sampling pdf is the Lambertian cos/pi, so every MIS pdf at an exit vertex
    is exactly the diffuse one and only the BSDF VALUE carries Ft."""
    return Float32(1.0) - fr_dielectric(abs(cos_theta), eta)


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

    The ordinary Lambertian scatter update, with one difference the harness
    establishes: the exit vertex's REVERSE density toward its predecessor is
    the hop's p_A (area measure, symmetric), not a direction pdf. With a
    cosine lobe cos_out/pdf_dir = pi."""
    var dvc_n = PI * (dvc * p_area + dvcm + mis_vm_weight_factor)
    var dvm_n = PI * (dvm * p_area + dvcm * mis_vc_weight_factor + Float32(1.0))
    var pdf_dir = cos_out / PI
    var dvcm_n = Float32(1.0) / pdf_dir if pdf_dir > Float32(1e-8) else Float32(0.0)
    return (dvcm_n, dvc_n, dvm_n)
