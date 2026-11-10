"""Jensen and Buhler 2002 inside SPPM: irradiance samples on translucent surfaces from SPPM's photons and NEE, and a
dipole integral of them at every camera hit on those surfaces. A biased reference for the random walk, not a rival."""
from std.math import floor, sqrt
from gonzales.core.geometry import Point3f, Vec3f, RGB, INV_PI, dot
from gonzales.core.rng import PCG32
from gonzales.core.spectrum import SampledWavelengths, SpectralSample, spectral_sample_to_rgb
from gonzales.scene.accel.bvh import SceneView
from gonzales.scene.appearance.materials import LobeKind, PhotonKind, sss_boundary_lobe
from gonzales.estimators.photon_grid import NEIGHBOUR_CELLS, neighbour_buckets, grid_weight, gather_disk_contains, gather_disk_coverage, disk_measure
from gonzales.estimators.dipole import dipole_of
from gonzales.estimators.irradiance_cache import IrradianceCache
from gonzales.integrators.sppm.sppm import SPPMPixel, SPPMPhoton, _sppm_nee_at


def jb_point_radiance(cache: IrradianceCache, i: Int, photons: Pointer[SPPMPhoton, MutUntrackedOrigin],
                      heads: Pointer[Int32, MutUntrackedOrigin], inv_cell: Float32, r2: Float32, ref sd: SceneView,
                      pass_wl: SampledWavelengths, pass_idx: Int, mut pcg: PCG32) -> SpectralSample:
    """Point i's radiance through its exit lobe this pass: arrival photons in a disk of radius sqrt(r2), plus NEE."""
    var p = cache.pos[unsafe_offset=i]
    var mi = cache.mat[unsafe_offset=i]
    var eta = cache.eta[unsafe_offset=i]
    return jb_point_photons(cache, i, photons, heads, inv_cell, r2, sd, pass_idx) \
        + _sppm_nee_at(_point_vp(p, cache.nrm[unsafe_offset=i], mi, eta, pass_wl), sd, pcg)


def jb_point_photons(cache: IrradianceCache, i: Int, photons: Pointer[SPPMPhoton, MutUntrackedOrigin],
                     heads: Pointer[Int32, MutUntrackedOrigin], inv_cell: Float32, r2: Float32, ref sd: SceneView,
                     pass_idx: Int) -> SpectralSample:
    """The photon half of jb_point_radiance."""
    var p = cache.pos[unsafe_offset=i]
    var n = cache.nrm[unsafe_offset=i]
    var eta = cache.eta[unsafe_offset=i]
    var phi = SpectralSample(Float32(0))
    var cix = Int(floor(p.x * inv_cell)); var ciy = Int(floor(p.y * inv_cell)); var ciz = Int(floor(p.z * inv_cell))
    var buckets = neighbour_buckets(cix, ciy, ciz)
    for b in range(NEIGHBOUR_CELLS):
        var h = buckets[b]
        if h < 0:
            break
        var bw = grid_weight(heads, h)
        var k = Int(heads[unsafe_offset=h])
        while k != -1:
            var ph = photons[unsafe_offset=k]
            var e = ph.pos - p
            var c = -dot(ph.dir_in.to_simd(), n.to_simd())
            if ph.is_volume == PhotonKind.surface and ph.curve < Int32(0) and c > Float32(0) \
                    and gather_disk_contains(e.to_simd(), e.length_sq(), r2, n.to_simd()):
                phi += ph.flux * (bw * sss_boundary_lobe(c, eta) * INV_PI)
            k = Int(ph.nxt)
    var cov = gather_disk_coverage(sd, p, n, sqrt(r2), salt=13 + pass_idx * 1000003)
    if cov > Float32(0):
        return phi * (Float32(1) / (cov * disk_measure(r2)))
    return SpectralSample(Float32(0))


def _point_vp(p: Point3f, n: Vec3f, mi: Int32, eta: Float32, wl: SampledWavelengths) -> SPPMPixel:
    """An irradiance point as a visible point with a walk exit's lobe, for SPPM's NEE."""
    return SPPMPixel(pos=p, normal=n, geo_normal=n, beta=RGB(Float32(1)), alb=RGB(Float32(1)), tau=RGB(Float32(0)),
                     N_acc=Float32(0), r2=Float32(0), valid=Int32(1), pidx=Int32(-1), is_volume=PhotonKind.surface,
                     ld=SpectralSample(Float32(0)), med_idx=Int32(-1), env=RGB(Float32(0)), gsum=RGB(Float32(0)),
                     coat_keep=Float32(1), mat_kind=LobeKind.sss_exit, wo=n, alpha=eta, mat_idx=mi,
                     hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0), wavelengths=wl)


def jb_dipole_vp(vps: Pointer[SPPMPixel, MutUntrackedOrigin], i: Int, cache: IrradianceCache, ref sd: SceneView,
                 pass_wl: SampledWavelengths):
    """A dipole visible point's radiance this pass, into its beta-weighted sum."""
    var vp = vps[unsafe_offset=i]
    if vp.valid == Int32(0) or vp.mat_kind != LobeKind.dipole:
        return
    var dip = dipole_of(sd.materials[unsafe_offset=Int(vp.mat_idx)], vp.alpha, pass_wl, sd.spectral)
    var l = cache.radiance(vp.pos, vp.mat_idx, dip)
    var h = sd.spectral
    var (r, g, b) = spectral_sample_to_rgb(h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, l, pass_wl)
    vps[unsafe_offset=i].gsum = vp.gsum + vp.beta * RGB(r, g, b)
