"""A medium as a shading point sees it: free flight, extinction and the phase function, in one place."""
from gonzales.core.geometry import Vec3f, dot
from gonzales.core.rng import PCG32
from gonzales.core.spectrum import SampledWavelengths, SpectralHandle, SpectralSample
from gonzales.scene.accel.bvh import SceneView
from gonzales.scene.appearance.media import (
    FreeFlight, Grid, Medium, NvdbGrid, hg_phase, hg_sample, medium_inv_sigma_t_spectral, medium_kernel_radius,
    medium_sigma_s_spectral, medium_sigma_t_at_spectral, medium_sigma_t_spectral, medium_transmittance_ratio_spectral,
    sample_free_flight, spectral_free_flight_weight, free_flight_mixture_pdf, medium_is_heterogeneous,
)


@fieldwise_init
struct MediumView(TrivialRegisterPassable):
    var med: Medium
    var grids: Pointer[Grid, MutUntrackedOrigin]
    var nvdb_grids: Pointer[NvdbGrid, MutUntrackedOrigin]
    var spectral: SpectralHandle

    @staticmethod
    @always_inline
    def at(ref sd: SceneView, idx: Int32) -> MediumView:
        return MediumView(sd.mediums[unsafe_offset=Int(idx)], sd.grids, sd.nvdbGrids, sd.spectral)

    @always_inline
    def free_flight(self, o: Vec3f, d: Vec3f, t_max: Float32, mut pcg: PCG32, wl: SampledWavelengths) -> FreeFlight:
        """A distance along the ray, for a light path: the camera's sampler (hero-lane mixture), without emission."""
        return self.free_flight_emitting(o, d, t_max, pcg, wl)

    @always_inline
    def ff_ratio(self, prev_vol: Bool, prev_sig_t: Float32, d: Float32, ff: FreeFlight, wl: SampledWavelengths) -> Float32:
        """dVC's free-flight factor on arrival: the other subpath's density of landing on the previous vertex over this
        flight's own. Both subpaths sample alike, so it is the lane mixture at the previous vertex (a collision when it
        scattered in the medium, a survival otherwise) over ff.pdf; heterogeneous media sample red alone."""
        if not medium_is_heterogeneous(self.med) and self.spectral.res > 0:
            return free_flight_mixture_pdf(self.sigma_t(wl), 4, d, prev_vol) / max(ff.pdf, Float32(1e-30))
        return (prev_sig_t if prev_vol else Float32(1)) / (ff.sig_t if ff.collided else Float32(1))

    @always_inline
    def free_flight_emitting(self, o: Vec3f, d: Vec3f, t_max: Float32, mut pcg: PCG32, wl: SampledWavelengths) -> FreeFlight:
        """A distance along the ray, for a camera path: `emission` holds what the segment emits."""
        var s = self.spectral
        return sample_free_flight(self.med, self.grids, self.nvdb_grids, o, d, t_max, pcg, wl, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65)

    @always_inline
    def weight(self, ff: FreeFlight, t_max: Float32, wl: SampledWavelengths) -> SpectralSample:
        var s = self.spectral
        return spectral_free_flight_weight(self.med, ff, t_max, wl, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65)

    @always_inline
    def transmittance_ratio(self, t: Float32, pdf: Float32, wl: SampledWavelengths) -> SpectralSample:
        var s = self.spectral
        return medium_transmittance_ratio_spectral(self.med, t, pdf, wl, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65)

    @always_inline
    def sigma_t(self, wl: SampledWavelengths) -> SpectralSample:
        """Per unit density."""
        var s = self.spectral
        return medium_sigma_t_spectral(self.med, wl, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65)

    @always_inline
    def sigma_s(self, wl: SampledWavelengths) -> SpectralSample:
        """Per unit density."""
        var s = self.spectral
        return medium_sigma_s_spectral(self.med, wl, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65)

    @always_inline
    def sigma_t_at(self, p: Vec3f, wl: SampledWavelengths) -> SpectralSample:
        var s = self.spectral
        return medium_sigma_t_at_spectral(self.med, self.grids, self.nvdb_grids, p, wl, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65)

    @always_inline
    def inv_sigma_t_at(self, p: Vec3f, wl: SampledWavelengths) -> SpectralSample:
        var s = self.spectral
        return medium_inv_sigma_t_spectral(self.med, self.grids, self.nvdb_grids, p, wl, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65)

    @always_inline
    def kernel_radius(self, p: Vec3f) -> Float32:
        return medium_kernel_radius(self.med, self.grids, self.nvdb_grids, p)

    @always_inline
    def phase(self, wo: Vec3f, wi: Vec3f) -> Float32:
        return hg_phase(dot(wo, wi), self.med.g)

    @always_inline
    def sample_phase(self, wo: Vec3f, u1: Float32, u2: Float32) -> SIMD[DType.float32, 4]:
        return hg_sample(wo, self.med.g, u1, u2)
