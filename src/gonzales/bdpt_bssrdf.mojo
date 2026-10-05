# BSSRDF exit sampling for BDPT/VCM subpaths.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import sqrt
from .geometry import RGB, Point3f, Vec3f, Frame, dot, PI
from .primitives import Ray, Intersection, PrimId, sphere_outward_normal
from .bssrdf import (
    dipole_max_radius, dipole_rd, dipole_mis_sigma_tr, dipole_sample_radius, bssrdf_probe_offset,
    bssrdf_exit_pdf_area, bssrdf_exit_ft,
)
from .bvh import SceneView, traverse_bvh2_core, test_spheres
from .rng import PCG32
from .sppm import _geom_normal

# The RGB siblings of these two (_eval_vertex / _eval_conductor_ggx) are gone:
# BDPT/VCM transport is spectral, so a connection multiplies two spectral
@fieldwise_init
struct BssrdfExitSample(TrivialRegisterPassable):
    """Result of sampling where a subsurface hop leaves the surface."""
    var ok: Bool
    var x_o: Point3f
    var n_o: Vec3f        # outward geometric normal at the exit
    var weight: RGB       # R_d(r) * Ft(entry) / p_A -- the exit lobe's Ft is NOT in here
    var p_area: Float32   # p_A(x_o | x_i), area measure; also the reverse density


def _bdpt_sample_bssrdf_exit(
    ref sd: SceneView,
    med_idx: Int,
    hit: Point3f,
    n_in: Vec3f,          # entry normal, facing the side the path arrived from
    cos_in: Float32,
    eta: Float32,
    mut pcg: PCG32,
) -> BssrdfExitSample:
    """ONE exit-point sampler for both VCM subpaths. The MIS weights assume the
    camera and light sides draw the hop from the same density (the hop is"""
    var med = sd.mediums[unsafe_offset=med_idx]
    var fail = BssrdfExitSample(False, hit, n_in, RGB(Float32(0)), Float32(0))
    var ft_in = bssrdf_exit_ft(cos_in, eta)
    if ft_in <= Float32(0.0):
        return fail
    var r_max = dipole_max_radius(med.sigma_s, med.sigma_a, med.g)
    var ch = Int(pcg.next_float() * Float32(3.0))
    if ch > 2: ch = 2
    var sigma_tr = dipole_mis_sigma_tr(med.sigma_s, med.sigma_a, med.g, ch)
    if sigma_tr <= Float32(0.0):
        return fail
    var r = dipole_sample_radius(sigma_tr, pcg.next_float())
    var phi = Float32(2.0) * PI * pcg.next_float()
    if r >= r_max:
        return fail
    var frm = Frame.from_z(n_in)
    var (off, seg_len) = bssrdf_probe_offset(
        r, phi, r_max, Vec3f(frm.x.x, frm.x.y, frm.x.z), Vec3f(frm.y.x, frm.y.y, frm.y.z), n_in)
    var probe_org = hit + off
    var probe_dir = n_in * Float32(-1.0)
    # Private probe slot: the caller's scratch still holds the intersection
    # the enclosing path loop is shading.
    var _probe_slot = Array[Intersection, 1](fill=Intersection(
        PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var probe_scratch = _probe_slot.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    probe_scratch[unsafe_offset=0].hit = Int8(0)
    var probe_ray = Ray(probe_org, probe_dir)
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe_ray, seg_len, probe_scratch,
                       sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    test_spheres(sd.spheres, Int(sd.sphereCount), probe_ray, probe_scratch)
    if probe_scratch[unsafe_offset=0].hit == Int8(0):
        return fail
    var pi = probe_scratch[unsafe_offset=0]
    # The exit must be on a subsurface boundary too, or the profile does not
    # describe what happens there.
    var pmat = sd.materials[unsafe_offset=Int(pi.primId.materialIndex)]
    if pmat.sss_boundary == Int8(0):
        return fail
    var x_o = probe_org + probe_dir * pi.tHit
    var n_o: Vec3f
    if pi.primId.type == Int8(4):
        n_o = sphere_outward_normal(x_o, sd.spheres[unsafe_offset=Int(pi.primId.id1)].center)
    else:
        n_o = _geom_normal(pi, sd.meshes, sd.instances, sd.spheres, x_o.to_simd())
    if dot(n_o, n_in) < Float32(0.0):
        n_o = n_o * Float32(-1.0)
    var d = x_o - hit
    var r_act = sqrt(dot(d, d))
    if r_act > r_max:
        return fail
    var p_area = bssrdf_exit_pdf_area(med.sigma_s, med.sigma_a, med.g, r_act, dot(n_o, n_in))
    if p_area <= Float32(1e-12):
        return fail
    var rd = dipole_rd(med.sigma_s, med.sigma_a, med.g, eta, r_act)
    var k = ft_in / p_area
    return BssrdfExitSample(True, x_o, n_o, RGB(rd.r * k, rd.g * k, rd.b * k), p_area)
