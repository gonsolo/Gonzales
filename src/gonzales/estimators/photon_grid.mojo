"""The thinned hash grid and the gather disk SPPM's photons and VCM's light vertices share."""
from std.math import sqrt, cos, sin, abs, min, max
from std.atomic import Atomic
from std.collections import Array
from std.memory import bitcast
from gonzales.core.geometry import Point3f, Vec3f, Ray, dot, PI
from gonzales.scene.shapes.primitives import Intersection, PrimId
from gonzales.scene.shapes.geom_normal import _geom_normal
from gonzales.scene.accel.bvh import SceneView, traverse_dispatch


comptime _HSIZE  = 1048576   # 2^20 hash buckets


@always_inline
def _hash_cell(ix: Int, iy: Int, iz: Int) -> Int:
    var h = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791
    return (h % _HSIZE + _HSIZE) % _HSIZE


# ── Thinned hash grid: SPPM's photons and VCM's light vertices ────────────────
# The bucket table is 2 * _HSIZE Int32s: heads[h] is bucket h's chain head,
# heads[_HSIZE + h] how many elements fell into it before thinning. A bucket
# holding n > cap of them keeps each with probability cap / n, and the gather
# scales every survivor by n / cap -- Russian roulette on photons, so the
# estimate's expectation is unchanged and only over-full buckets pay variance.
#
# Prior art, photon mapping: Hachisuka & Jensen 2010's "stochastic hashing"
# (cap 1), made unbiased by Davidovic et al. 2014 ("rectified stochastic hash
# grid", TOG 33(3) s6.2), and Kern et al. 2023 (JCGT 12(1) s4.1), which
# reservoir-samples cap photons per cell and weights n/cap. This is the
# Bernoulli form of the same idea.
#
# Why: every path from one tiny bright emitter piles into a handful of cells.
# barcelona-pavilion-night's candle lanterns and small sphere lights put up to
# 283K photons in one cell (the median cell holds 3), and walking those chains
# at every nearby gather point was over 90% of the render for BOTH photon
# integrators: VCM 745 -> 134 s with the cap; SPPM's gather was ~12 s of every
# ~9 s-per-pass budget (under ncu) before it got the same cap. VCM's sweep
# (320x170, 16 spp) took 37 / 18 / 13 / 12 / 11.5 s for none / 1024 / 256 /
# 64 / 16, and the noise thinning added at 256 was ~7% of the render's own
# seed-to-seed noise in the lantern zone.
comptime _PHOTON_BUCKET_CAP = Int32(4096)


@always_inline
def grid_reset_cell(heads: Pointer[Int32, MutUntrackedOrigin], h: Int):
    heads[unsafe_offset=h] = Int32(-1)
    heads[unsafe_offset=_HSIZE + h] = Int32(0)


@always_inline
def grid_count(heads: Pointer[Int32, MutUntrackedOrigin], h: Int):
    """First pass of a thinned grid build: tally one element into bucket h."""
    _ = Atomic.fetch_add(heads.unsafe_offset(_HSIZE + h), Int32(1))


@always_inline
def grid_keep(heads: Pointer[Int32, MutUntrackedOrigin], h: Int, salt: Int, pos: Point3f,
              cap: Int32 = _PHOTON_BUCKET_CAP) -> Bool:
    """Whether the element at `pos` survives thinning of bucket h (probability
    cap / n). The coin hashes `salt` and the position, so it is blind to the
    element's flux. `salt` must be DETERMINISTIC and change every pass: VCM
    passes its light-vertex slot (fixed per light path), SPPM the pass index
    -- an SPPM photon's slot comes from an atomic fetch-add, whose order races,
    and salting with it made every --sppm render differ run to run."""
    var n = heads[unsafe_offset=_HSIZE + h]
    if n <= cap:
        return True
    return UInt64(grid_coin_bits(salt, pos)) * UInt64(n) < UInt64(cap) << 32


@always_inline
def grid_coin_bits(salt: Int, pos: Point3f) -> UInt32:
    """The thinning coin as 32 uniform bits: a splitmix hash of `salt` and
    the position (see grid_keep for the salt rules)."""
    var bits = bitcast[DType.uint32, 4](SIMD[DType.float32, 4](pos.x, pos.y, pos.z, Float32(0)))
    var z = UInt64(salt) * UInt64(0x9E3779B97F4A7C15) ^ (UInt64(bits[0]) << 32 | UInt64(bits[1])) ^ UInt64(bits[2])
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    z ^= z >> 31
    return UInt32(z >> 32)


@always_inline
def grid_push[use_gpu: Bool](heads: Pointer[Int32, MutUntrackedOrigin], h: Int, k: Int) -> Int32:
    """Make element `k` bucket h's chain head; returns the old head, which the
    caller stores as k's link. Atomic exchange for racing GPU threads (and
    parallel CPU workers), a plain read-modify-write for a serial loop."""
    comptime if use_gpu:
        return Atomic._xchg(heads.unsafe_offset(h), Int32(k))
    else:
        var old = heads[unsafe_offset=h]
        heads[unsafe_offset=h] = Int32(k)
        return old


@always_inline
def grid_weight(heads: Pointer[Int32, MutUntrackedOrigin], h: Int,
                cap: Int32 = _PHOTON_BUCKET_CAP) -> Float32:
    """1 / keep-probability of bucket h's surviving elements."""
    var n = heads[unsafe_offset=_HSIZE + h]
    if n <= cap:
        return Float32(1)
    return Float32(n) / Float32(cap)


@always_inline
def disk_measure(r2: Float32) -> Float32:
    return PI * r2


@always_inline
def ball_measure(r2: Float32) -> Float32:
    return (Float32(4.0) / Float32(3.0)) * PI * r2 * sqrt(r2)


@always_inline
def ribbon_measure(r2: Float32, width: Float32) -> Float32:
    """A fibre within r, a ribbon of its own width facing every direction."""
    return Float32(2) * sqrt(r2) * width


@always_inline
def gather_disk_contains(e: Vec3f, dist2: Float32, r2: Float32, n: Vec3f) -> Bool:
    """Does a photon at offset `e` from a gather point lie inside the gather
    DISK -- radius sqrt(r2), on the tangent plane with normal `n` -- rather
    than merely inside the BALL of that radius?

    THE photon-acceptance test for a surface density estimate, shared by
    SPPM's gather and VCM's merge. A surface estimate divides the photons it
    finds by one disk's area, pi r^2, so it is only right if the photons came
    from that disk. A ball also sweeps in photons lying on OTHER surfaces
    within reach -- an adjacent wall, a step, the parallel surface below a
    shelf -- and sums them against the same single disk. VCM found this first
    (classroom read 2.5x pbrt with merging on, 0.985x with it off) and grew
    this guard; SPPM never did, and its gather radius in a room-sized scene is
    large (0.51 scene units in barcelona-pavilion, around its pool). A photon
    on this surface sits on the tangent plane to float precision, so a tenth
    of the radius is generous."""
    if dist2 > r2:
        return False
    var off = dot(e, n)
    return off * off <= r2 * Float32(0.01)


comptime GATHER_BALL_PROBES = 16


def gather_ball_reach[probes: Int = GATHER_BALL_PROBES](ref sd: SceneView, p: Point3f, r: Float32, salt: Int = 11) -> Float32:
    """How far the gather BALL around a volume point p may reach, at most r: the distance to the nearest surface. A
    dense medium keeps its visible points within a mean free path of the boundary; a ball reaching past it finds no
    photons there and reads dark (a skin ball at sigma_t 40 lost 9%), and dividing by the covered fraction instead
    reads bright, since the density climbs with depth. A ball that stays inside is symmetric about p, so a gradient
    cancels. `probes` rays out of p on a spherical Fibonacci set that a hash of `salt` and p rotates at random
    (uniform axis, uniform spin); 16 overshoot a plane's distance by under 4%."""
    var _local = Array[Intersection, 1](fill=Intersection(
        PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var mem = _local.unsafe_ptr()   # tracked: keeps _local alive while read
    var rot = Float32(grid_coin_bits(salt, p)) * Float32(2.3283064e-10) * Float32(6.2831853)
    var az = Float32(1) - Float32(2) * Float32(grid_coin_bits(salt + 1, p)) * Float32(2.3283064e-10)
    var aphi = Float32(grid_coin_bits(salt + 2, p)) * Float32(2.3283064e-10) * Float32(6.2831853)
    var ar = sqrt(max(Float32(1) - az * az, Float32(0)))
    var n = Vec3f(ar * cos(aphi), ar * sin(aphi), az)
    var sgn = Float32(1) if n.z >= Float32(0) else Float32(-1)
    var ka = Float32(-1) / (sgn + n.z)
    var kb = n.x * n.y * ka
    var t1 = Vec3f(Float32(1) + sgn * n.x * n.x * ka, sgn * kb, -sgn * n.x)
    var t2 = Vec3f(kb, sgn + n.y * n.y * ka, -n.y)
    var reach = r
    comptime for i in range(probes):
        var z = Float32(1) - (Float32(2 * i) + Float32(1)) / Float32(probes)
        var rho = sqrt(max(Float32(1) - z * z, Float32(0)))
        var ang = rot + Float32(i) * Float32(2.39996323)
        var c = rho * cos(ang)
        var sn = rho * sin(ang)
        var ray = Ray(p, Vec3f(t1.x * c + t2.x * sn + n.x * z, t1.y * c + t2.y * sn + n.y * z, t1.z * c + t2.z * sn + n.z * z))
        mem[unsafe_offset=0].hit = Int8(0)
        traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, reach, mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        if mem[unsafe_offset=0].hit != Int8(0) and mem[unsafe_offset=0].primId.type != Int8(5):
            reach = min(reach, mem[unsafe_offset=0].tHit)
    return reach


comptime GATHER_COVERAGE_PROBES = 8


def gather_disk_coverage[probes: Int = GATHER_COVERAGE_PROBES](
    ref sd: SceneView, p: Point3f, n: Vec3f, r: Float32, salt: Int = 7,
) -> Float32:
    """The fraction of the gather disk -- radius r on the tangent plane of the
    geometric normal n at p -- over which a surface lies that
    gather_disk_contains would take photons from.

    A surface density estimate divides the photons it finds by the disk's
    area, so it assumes the whole disk is surface. Where the disk hangs off a
    small object -- a lantern panel, a frame, a leaf -- the photons can only
    come from the covered part, and dividing by all of pi r^2 reads low by
    exactly this fraction. That bias is first order in r: on
    barcelona-pavilion at night VCM lost ~30% of the candle lanterns' light
    at its default radius, twice that at double the radius, and merging
    switched off matched the path tracer. Dividing by the coverage removes
    it at any radius (lanterns 0.70 -> 0.91, 0.85 -> 0.99 of the reference).

    Estimated with `probes` short rays down the normal through
    gather_disk_contains's +-0.1 r slab, at stratified points of a
    golden-angle spiral that a hash of `salt` and p rotates. A hit counts
    when its geometric normal is within the merge's 0.7 cosine of n (either
    side: a thin panel is surface on both). Curves are not surfaces here, so
    a hair query should not call this.

    `salt` should vary from call to call for a QUERY POINT THAT STAYS FIXED
    across many calls -- same rule grid_keep's own salt already documents,
    and SPPM's visible points (unlike VCM's re-jittered camera vertices) are
    generated ONCE and gathered every one of its dozens of passes at the SAME
    vp.pos with a merely-shrinking radius, so a frozen salt gives every pass
    the identical probe rays. SPPM passes a per-pass salt (its pass index)
    for exactly this reason -- but TESTED AND RULED OUT as the explanation
    for its residual bias vs. a fixed salt (barcelona-pavilion night, whole
    image 0.998 either way; the 99-99.9% bin 1.054 fixed-salt vs 1.052
    per-pass, within noise of 2 seeds). Kept anyway as the more defensible
    default; the actual residual (SPPM whole 0.929->0.998 clean, but its
    99-99.9% bin overshoots to ~1.05 while its 99.9-100% bin stays ~8% low,
    where VCM's equivalent correction lands both within 1%) is still open.
    Leading unexplored
    hypothesis: SPPM's per-VP progressive radius (Knaus-Zwicker, this
    file's _ALPHA/ratio update) shrinks SLOWER for a VP whose disk is
    truncated (fewer accepted photons -> `ratio` closer to 1 -> less
    shrinkage), a feedback the coverage correction does not touch (it
    rescales `phi`, not the `M` count `ratio` is computed from) and that
    VCM's single global per-pass radius has no analogue of."""
    var sgn = Float32(1) if n.z >= Float32(0) else Float32(-1)
    var a = Float32(-1) / (sgn + n.z)
    var b = n.x * n.y * a
    var t1 = Vec3f(Float32(1) + sgn * n.x * n.x * a, sgn * b, -sgn * n.x)
    var t2 = Vec3f(b, sgn + n.y * n.y * a, -n.y)
    # A private slot, as _visible_transmittance: the caller's scratch may be
    # live in an enclosing traversal.
    var _local = Array[Intersection, 1](fill=Intersection(
        PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var mem = _local.unsafe_ptr()   # tracked: keeps _local alive while read
    var rot = Float32(grid_coin_bits(salt, p)) * Float32(2.3283064e-10) * Float32(6.2831853)
    var lift = Float32(0.1) * r
    var hits = 0
    comptime for i in range(probes):
        var rad = r * sqrt((Float32(i) + Float32(0.5)) / Float32(probes))
        var ang = rot + Float32(i) * Float32(2.39996323)
        var ca = cos(ang) * rad
        var sa = sin(ang) * rad
        var org = Point3f(p.x + t1.x * ca + t2.x * sa + n.x * lift,
                          p.y + t1.y * ca + t2.y * sa + n.y * lift,
                          p.z + t1.z * ca + t2.z * sa + n.z * lift)
        var ray = Ray(org, Vec3f(-n.x, -n.y, -n.z))
        mem[unsafe_offset=0].hit = Int8(0)
        traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(2) * lift, mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        if mem[unsafe_offset=0].hit != Int8(0) and mem[unsafe_offset=0].primId.type != Int8(4):
            var ng = _geom_normal(mem[unsafe_offset=0], sd.meshes, sd.instances)
            if abs(dot(ng, n)) > Float32(0.7):
                hits += 1
    return Float32(hits) / Float32(probes)
