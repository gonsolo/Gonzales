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

from std.math import exp, sqrt, max, min, log
from gonzales.geometry import RGB


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
