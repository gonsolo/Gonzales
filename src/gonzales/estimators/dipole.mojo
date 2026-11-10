"""The classical dipole diffusion BSSRDF (Jensen et al. 2001), as Jensen and Buhler 2002 integrate it."""
from std.math import exp, max, sqrt
from gonzales.core.geometry import PI
from gonzales.scene.appearance.materials import Material
from gonzales.core.spectrum import SampledWavelengths, SpectralHandle, SpectralSample, spec_refl, spec_refl_unbounded

comptime F4 = SIMD[DType.float32, 4]


@always_inline
def diffuse_fresnel_reflectance(eta: Float32) -> Float32:
    """Egan and Hilgeman's fit of the hemispherical Fresnel reflectance seen from inside, eta = inside / outside."""
    if eta < Float32(1):
        return Float32(-0.4399) + Float32(0.7099) / eta - Float32(0.3319) / (eta * eta) + Float32(0.0636) / (eta * eta * eta)
    return Float32(-1.440) / (eta * eta) + Float32(0.710) / eta + Float32(0.668) + Float32(0.0636) * eta


@fieldwise_init
struct Dipole(TrivialRegisterPassable):
    """A semi-infinite medium's diffuse reflectance R_d(r) at four wavelengths."""
    var zr: F4
    var zv: F4
    var sigma_tr: F4
    var alpha_p: F4

    @staticmethod
    def of(sigma_a: F4, sigma_sp: F4, eta: Float32) -> Dipole:
        var st = max(sigma_a + sigma_sp, F4(1e-12))
        var fdr = diffuse_fresnel_reflectance(eta)
        var a = (Float32(1) + fdr) / (Float32(1) - fdr)
        var zr = F4(1) / st
        return Dipole(zr, zr * (Float32(1) + Float32(4) / Float32(3) * a), sqrt(Float32(3) * sigma_a * st), sigma_sp / st)

    @always_inline
    def rd(self, r2: Float32) -> SpectralSample:
        """Radiant exitance at distance sqrt(r2) per unit incident flux."""
        var dr = sqrt(r2 + self.zr * self.zr)
        var dv = sqrt(r2 + self.zv * self.zv)
        var t = self.zr * (Float32(1) + self.sigma_tr * dr) * exp(-self.sigma_tr * dr) / (dr * dr * dr) \
              + self.zv * (Float32(1) + self.sigma_tr * dv) * exp(-self.sigma_tr * dv) / (dv * dv * dv)
        var v = self.alpha_p * t * (Float32(1) / (Float32(4) * PI))
        return SpectralSample(v[0], v[1], v[2], v[3])


def dipole_of(mat: Material, eta: Float32, wl: SampledWavelengths, h: SpectralHandle) -> Dipole:
    """A pbrt subsurface material's dipole, from the coefficients its random walk upsamples (sss_walk_start); eta is
    the boundary's index relative to the medium outside."""
    var c1 = mat.checker_tex1
    var c2 = mat.checker_tex2
    var st = spec_refl_unbounded(h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, c1.r, c1.g, c1.b, wl)
    var al = spec_refl(h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, c2.r, c2.g, c2.b, wl)
    var st4 = F4(st.v0, st.v1, st.v2, st.v3)
    var ss4 = F4(al.v0, al.v1, al.v2, al.v3) * st4
    return Dipole.of(st4 - ss4, ss4 * (Float32(1) - mat.checker_uscale), eta)
