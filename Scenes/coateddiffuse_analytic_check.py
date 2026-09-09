#!/usr/bin/env python3
"""Independent analytic ground truth for Scenes/coateddiffuse-grazing-probe.pbrt.

The probe is the one configuration where a smooth-coat-over-Lambertian layered
BSDF has a closed form: smooth coat (roughness 0), eta 1.5, Lambertian base,
and a DISTANT light at normal incidence, so there is no microfacet lobe, no
spatial variation, and no visibility term.

    f(wi,wo) = rho * (1-F(cos_i)) * (1-F(cos_o))
               / ( pi * eta^2 * (1 - rho*F_di) )

F_di is the *internal* diffuse Fresnel reflectance -- the cosine-weighted
hemispherical average of the Fresnel reflectance seen from inside the coat.
The 1/(1 - rho*F_di) factor sums the whole TIR recycling series, and 1/eta^2 is
the radiance compression on exit. This is the closed form documented in
docs/05_reflection_models.md (Mitsuba's `plastic`); it assumes light
re-randomises to a cosine distribution on each internal bounce, which is exact
here because the base IS Lambertian.

Crucially this must be evaluated PER PIXEL, not once at the image centre: at
grazing elevations F(cos_o) varies steeply across the patch, so the mean of f
is not f of the mean angle. That distinction is worth ~7% at 4 degrees.

Run from the repo root:  python3 Scenes/coateddiffuse_analytic_check.py
"""
import math
import subprocess

import numpy as np
import OpenImageIO as oiio

PBRT = "/home/gonsolo/src/pbrt-v4/gonsolo/pbrt"
SCENE = "Scenes/coateddiffuse-grazing-probe.pbrt"
OUT = "coateddiffuse-grazing-probe.exr"
DIST = 6.0
RES = 120
FOV = 20.0
RHO = 0.5
ETA = 1.5
L_DISTANT = 10.0
ELEVATIONS = (60.0, 30.0, 15.0, 8.0, 4.0)


def fresnel_dielectric(cos_i, eta):
    """Unpolarised Fresnel reflectance, air -> medium of index `eta`."""
    cos_i = np.clip(cos_i, 0.0, 1.0)
    sin2_t = (1.0 - cos_i * cos_i) / (eta * eta)
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
    r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
    r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
    return 0.5 * (r_parl * r_parl + r_perp * r_perp)


def fresnel_internal_diffuse(eta):
    """Cosine-weighted hemispherical average of Fresnel seen from INSIDE.

    Integrated numerically rather than using the Egan-Hilgeman polynomial fit,
    so this stays an independent reference rather than inheriting someone
    else's approximation. Angles past the critical angle are total internal
    reflection and contribute 1.
    """
    n = 200000
    # cosine-weighted sampling of the inner hemisphere: cos = sqrt(u)
    u = (np.arange(n) + 0.5) / n
    cos_in = np.sqrt(u)
    sin_in = np.sqrt(np.maximum(0.0, 1.0 - cos_in * cos_in))
    sin_t = sin_in * eta          # leaving dense -> thin
    tir = sin_t >= 1.0
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(sin_t, 1.0) ** 2))
    # Fresnel is reciprocal: evaluate with the roles swapped for the transmitted side.
    r_parl = (cos_in - eta * cos_t) / (cos_in + eta * cos_t)
    r_perp = (eta * cos_in - cos_t) / (eta * cos_in + cos_t)
    f = 0.5 * (r_parl * r_parl + r_perp * r_perp)
    f = np.where(tir, 1.0, f)
    return float(f.mean())


def view_cosines(elev_deg):
    """cos(theta_o) at every pixel of the central half, from the real camera."""
    a = math.radians(elev_deg)
    eye = np.array([0.0, DIST * math.sin(a), DIST * math.cos(a)])
    fwd = -eye / np.linalg.norm(eye)
    up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    # pbrt maps fov to the SHORTER axis; the image is square here.
    t = math.tan(math.radians(FOV) / 2.0)
    idx = (np.arange(RES) + 0.5) / RES * 2.0 - 1.0     # -1..1, pixel centres
    sx, sy = np.meshgrid(idx * t, -idx * t)            # +y is up in pbrt raster
    d = (fwd[None, None, :]
         + sx[..., None] * right[None, None, :]
         + sy[..., None] * up[None, None, :])
    d /= np.linalg.norm(d, axis=2, keepdims=True)

    lo, hi = RES // 4, 3 * RES // 4                    # same crop as the sweep
    d = d[lo:hi, lo:hi]
    # Plane is y=0; the camera looks down, so d_y < 0 for pixels that hit it.
    return np.abs(d[:, :, 1])


def analytic_mean(elev_deg, f_di):
    cos_o = view_cosines(elev_deg)
    f_o = fresnel_dielectric(cos_o, ETA)
    f_i = fresnel_dielectric(1.0, ETA)                 # distant light, normal incidence
    f = (RHO * (1.0 - f_i) * (1.0 - f_o)
         / (math.pi * ETA * ETA * (1.0 - RHO * f_di)))
    # Distant light of radiance L delivers irradiance L*cos_i = L*1 here.
    return float((f * L_DISTANT).mean())


def read_center(path):
    buf = oiio.ImageBuf(path)
    if buf.has_error:
        raise RuntimeError(f"{path}: {buf.geterror()}")
    a = np.array(buf.get_pixels(oiio.FLOAT))[:, :, :3]
    h, w, _ = a.shape
    return a[h // 4: 3 * h // 4, w // 4: 3 * w // 4]


def write_scene(elev_deg):
    src = open(SCENE).read()
    a = math.radians(elev_deg)
    src = src.replace("EYEY", f"{DIST * math.sin(a):.6f}")
    src = src.replace("EYEZ", f"{DIST * math.cos(a):.6f}")
    tmp = "/tmp/g2/analytic_tmp.pbrt"
    open(tmp, "w").write(src)
    return tmp


def main():
    f_di = fresnel_internal_diffuse(ETA)
    print(f"internal diffuse Fresnel F_di(eta={ETA}) = {f_di:.5f}")
    print(f"{'elev':>6} {'analytic':>10} {'gonzales':>10} {'pbrt':>10} "
          f"{'g/ana':>8} {'pbrt/ana':>9}")
    for elev in ELEVATIONS:
        tmp = write_scene(elev)
        subprocess.run(["./build/gonzales", "--spp", "512", "--no-denoise", tmp],
                       check=True, capture_output=True)
        g = read_center(OUT).mean()
        subprocess.run([PBRT, "--outfile", "/tmp/g2/ana_pbrt.exr",
                        "--spp", "512", tmp], check=True, capture_output=True)
        p = read_center("/tmp/g2/ana_pbrt.exr").mean()
        a = analytic_mean(elev, f_di)
        print(f"{elev:6.0f} {a:10.5f} {g:10.5f} {p:10.5f} "
              f"{g / a:8.4f} {p / a:9.4f}")


if __name__ == "__main__":
    main()
