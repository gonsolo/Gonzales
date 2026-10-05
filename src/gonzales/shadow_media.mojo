# Transmittance of a surface-NEE shadow segment through interface-bounded media.
from std.collections import Array
from std.math import exp, log, max, abs
from .geometry import Point3f, Vec3f, dot
from .materials import Material, MatKind
from .primitives import Ray, Intersection, PrimId, TriangleMesh, Sphere, Instance
from .curves import Curve
from .media import Medium, MediumInterface, Grid, NvdbGrid, MEDIUM_TRACK_MAX_ITERS, medium_sigma_t_spectral, grid_sample_density, nvdb_sample_density
from .bvh import BVH2Node, traverse_bvh2_core, test_spheres
from .geom_normal import _geom_normal
from .rng import PCG32
from .spectrum import SpectralHandle, SampledWavelengths, SpectralSample


@always_inline
def segment_transmittance(
    med: Medium, org: Vec3f, dir: Vec3f, t: Float32,
    grids: Pointer[Grid, MutUntrackedOrigin], nvdb_grids: Pointer[NvdbGrid, MutUntrackedOrigin],
    spectral: SpectralHandle, wl: SampledWavelengths, mut pcg: PCG32,
) -> SpectralSample:
    var st = medium_sigma_t_spectral(med, wl, spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y, spectral.cie_z, spectral.d65)
    var use_nvdb = med.nvdb_idx >= Int32(0)
    if not use_nvdb and med.grid_idx < Int32(0):
        return SpectralSample(exp(-st.v0 * t), exp(-st.v1 * t), exp(-st.v2 * t), exp(-st.v3 * t))
    # Ratio tracking against the grid's majorant, all four hero lanes on one track.
    var majorant = nvdb_grids[unsafe_offset=Int(med.nvdb_idx)].max_density if use_nvdb else grids[unsafe_offset=Int(med.grid_idx)].max_density
    var sigma_maj = majorant * max(max(st.v0, st.v1), max(st.v2, st.v3))
    var T = SpectralSample(Float32(1.0))
    if sigma_maj <= Float32(0.0):
        return T
    var ts = Float32(0.0)
    for _ in range(MEDIUM_TRACK_MAX_ITERS):
        ts += -log(max(pcg.next_float(), Float32(1e-7))) / sigma_maj
        if ts >= t:
            break
        var p = org + dir * ts
        var density = nvdb_sample_density(nvdb_grids[unsafe_offset=Int(med.nvdb_idx)], p) if use_nvdb else grid_sample_density(grids[unsafe_offset=Int(med.grid_idx)], p)
        var k = density / sigma_maj
        T = T * SpectralSample(Float32(1.0) - k * st.v0, Float32(1.0) - k * st.v1, Float32(1.0) - k * st.v2, Float32(1.0) - k * st.v3)
        if max(max(T.v0, T.v1), max(T.v2, T.v3)) < Float32(1e-4):
            return SpectralSample(Float32(0.0))
    return T


def shadow_transmittance(
    org: Vec3f, dir: Vec3f, tmax: Float32, start_med: Int32,
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin], primIds: Pointer[PrimId, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin],
    spheres: Pointer[Sphere, MutUntrackedOrigin], n_spheres: Int,
    materials: Pointer[Material, MutUntrackedOrigin],
    mediums: Pointer[Medium, MutUntrackedOrigin], medium_ifaces: Pointer[MediumInterface, MutUntrackedOrigin],
    grids: Pointer[Grid, MutUntrackedOrigin], nvdb_grids: Pointer[NvdbGrid, MutUntrackedOrigin],
    spectral: SpectralHandle, wl: SampledWavelengths, mut pcg: PCG32,
) -> SpectralSample:
    """Walk the segment through interface surfaces, tracking the current medium; black if an opaque surface blocks it."""
    var T = SpectralSample(Float32(1.0))
    var cur = start_med
    var o = org
    var remaining = tmax
    var _local = Array[Intersection, 1](fill=Intersection(
        PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var inter_mem = _local.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for _ in range(8):
        if remaining < Float32(1e-4):
            break
        var ray = Ray(Point3f(o[0], o[1], o[2]), dir)
        inter_mem[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, remaining, inter_mem, blasNodesArr, blasPrimIdsArr, instances)
        # Analytic spheres are not in the BVH: test them against the sentinel hit at `remaining`.
        var had_bvh_hit = inter_mem[unsafe_offset=0].hit != Int8(0)
        if not had_bvh_hit:
            inter_mem[unsafe_offset=0].hit = Int8(1)
            inter_mem[unsafe_offset=0].tHit = remaining
            inter_mem[unsafe_offset=0].primId.type = Int8(0)
        test_spheres(spheres, n_spheres, ray, inter_mem)
        if not had_bvh_hit and inter_mem[unsafe_offset=0].primId.type != Int8(4):
            inter_mem[unsafe_offset=0].hit = Int8(0)
        var hit = inter_mem[unsafe_offset=0].hit != Int8(0)
        var t_seg = inter_mem[unsafe_offset=0].tHit if hit else remaining
        if Int(cur) >= 0:
            T = T * segment_transmittance(mediums[unsafe_offset=Int(cur)], o, dir, t_seg, grids, nvdb_grids, spectral, wl, pcg)
            if T.is_black():
                return T
        if not hit:
            break
        var inter = inter_mem[unsafe_offset=0]
        var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
        if mat.type != MatKind.interface:
            return SpectralSample(Float32(0.0))
        var p = o + dir * t_seg
        if mat.medium_interface_idx >= Int32(0):
            var iface = medium_ifaces[unsafe_offset=Int(mat.medium_interface_idx)]
            var n = _geom_normal(inter, meshes, instances, spheres, p)
            cur = iface.outside_medium_idx if dot(dir, n) > Float32(0.0) else iface.inside_medium_idx
        o = p + dir * Float32(0.0002)
        remaining = remaining - t_seg - Float32(0.0002)
    return T
