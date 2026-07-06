# Unit tests for the device-portable helper functions extracted from
# sppm.mojo (and shared with bdpt.mojo / gpu_sppm.mojo): normal computation,
# cosine-hemisphere sampling, homogeneous free-flight sampling, uniform
# area-light sampling, the photon-grid spatial hash, and the dielectric
# bounce helper.

from std.math import abs, sqrt, log, exp
from std.memory import alloc
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import (
    RGB, Point3f, Vec3f, PrimId_C, Intersection_C, TriangleMesh_C,
    AreaLight_C, Medium_C, dot, cross, fr_dielectric, reflect, PI,
)
from gonzales.sppm import (
    _hash_cell, _cosine_hemisphere_sample, sample_homogeneous_free_flight,
    sample_area_light_uniform, _geom_normal, _shading_normal_at, _dielectric_bounce,
    HomogeneousFreeFlight, AreaLightSample, _HSIZE,
)
from gonzales.rng import PCG32

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _simd_close(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> Bool:
    return _close(a[0], b[0]) and _close(a[1], b[1]) and _close(a[2], b[2])

def _simd_len(v: SIMD[DType.float32, 3]) -> Float32:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

# ── _hash_cell ────────────────────────────────────────────────────────────────

def test_hash_cell_same_input_is_deterministic() raises:
    """Same (ix,iy,iz) must always map to the same bucket — the whole grid
    (insert then gather over 3x3x3 neighborhoods) relies on this."""
    assert_true(_hash_cell(3, -7, 42) == _hash_cell(3, -7, 42))
    assert_true(_hash_cell(0, 0, 0) == _hash_cell(0, 0, 0))

def test_hash_cell_differs_across_a_few_cells() raises:
    """Weak collision sanity (not a cryptographic claim): a handful of
    distinct cell coordinates should not all collapse onto one bucket."""
    var h0 = _hash_cell(0, 0, 0)
    var h1 = _hash_cell(1, 0, 0)
    var h2 = _hash_cell(0, 1, 0)
    var h3 = _hash_cell(0, 0, 1)
    var h4 = _hash_cell(5, -3, 11)
    assert_true(h0 != h1)
    assert_true(h0 != h2)
    assert_true(h0 != h3)
    assert_true(h0 != h4)

def test_hash_cell_always_in_range() raises:
    """_build_grid/_gather_update index a fixed-size `heads` array of
    _HSIZE buckets directly with this result — it must never fall outside
    [0, _HSIZE), including for negative cell coordinates (floor() of a
    negative photon position)."""
    for coords in [(0, 0, 0), (-1, -1, -1), (100, -200, 300), (-999999, 5, -5)]:
        var h = _hash_cell(coords[0], coords[1], coords[2])
        assert_true(h >= 0)
        assert_true(h < _HSIZE)

# ── _cosine_hemisphere_sample ────────────────────────────────────────────────
# Frisvad-frame construction: d = tangent*lx + bitangent*lz + n*ly, with
# ly = sqrt(max(0, 1-u1)). Since tangent/bitangent are unit and orthogonal to
# n, dot(d, n) == ly exactly — a closed form independent of u2 and of the
# choice of normal.

def test_cosine_hemisphere_sample_cos_theta_matches_sqrt_one_minus_u1() raises:
    for n in [
        SIMD[DType.float32, 3](0.0, 0.0, 1.0),
        SIMD[DType.float32, 3](0.0, 0.0, -1.0),
        SIMD[DType.float32, 3](0.0, 1.0, 0.0),
        SIMD[DType.float32, 3](0.267261, 0.534522, 0.801784),  # normalize(1,2,3)
    ]:
        for u1 in [Float32(0.0), Float32(0.3), Float32(0.7), Float32(0.999)]:
            var d = _cosine_hemisphere_sample(n, u1, Float32(0.42))
            var cos_theta = dot(d, n)
            assert_true(_close(cos_theta, sqrt(Float32(1.0) - u1)))

def test_cosine_hemisphere_sample_u1_zero_returns_the_normal_itself() raises:
    """At u1=0, lx=lz=0 and ly=1, so the formula collapses to d == n exactly
    — regardless of u2 (the azimuthal angle is undefined when there's no
    in-plane component)."""
    var n = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
    for u2 in [Float32(0.0), Float32(0.37), Float32(0.9)]:
        var d = _cosine_hemisphere_sample(n, Float32(0.0), u2)
        assert_true(_simd_close(d, n))

def test_cosine_hemisphere_sample_is_always_unit_length() raises:
    var n = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    for u1 in [Float32(0.01), Float32(0.5), Float32(0.99)]:
        for u2 in [Float32(0.05), Float32(0.5), Float32(0.95)]:
            var d = _cosine_hemisphere_sample(n, u1, u2)
            assert_true(_close(_simd_len(d), Float32(1.0)))

def test_cosine_hemisphere_sample_stays_in_hemisphere() raises:
    var n = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    for u1 in [Float32(0.0), Float32(0.2), Float32(0.6), Float32(0.999)]:
        for u2 in [Float32(0.1), Float32(0.5), Float32(0.9)]:
            var d = _cosine_hemisphere_sample(n, u1, u2)
            assert_true(dot(d, n) >= Float32(0.0))

# ── sample_homogeneous_free_flight ───────────────────────────────────────────

def _medium(sigma_a: RGB, sigma_s: RGB) -> Medium_C:
    return Medium_C(sigma_a, sigma_s, Float32(0.0), Int32(-1), Float32(0.0), Float32(0.0))

def test_free_flight_zero_extinction_never_collides_and_keeps_t_surf() raises:
    """Sigma_t <= 0 is the medium's early-out branch: no collision is
    possible, t_free is defined to just be whatever t_surf was passed in,
    and the transmittance over the (vacuum) segment is exactly 1."""
    var med = _medium(RGB(Float32(0.0)), RGB(Float32(0.0)))
    var pcg = PCG32(UInt64(123), UInt64(1))
    var ff = sample_homogeneous_free_flight(med, Float32(17.0), pcg)
    assert_false(ff.collided)
    assert_true(_close(ff.t_free, Float32(17.0)))
    assert_true(_close(ff.transmittance.r, Float32(1.0)))
    assert_true(_close(ff.transmittance.g, Float32(1.0)))
    assert_true(_close(ff.transmittance.b, Float32(1.0)))

def test_free_flight_t_free_matches_closed_form_inversion_formula() raises:
    """T_free must equal -ln(max(u, 1e-7))/sigma_t for the exact `u` the
    function draws from `pcg` — verified by peeking that same draw from an
    identically-seeded generator before feeding a fresh copy to the
    function under test."""
    var seed = UInt64(555)
    var seq = UInt64(9)
    var peek = PCG32(seed, seq)
    var u = peek.next_float()
    var sigma_t = Float32(2.5)
    var expected_t_free = -log(max(u, Float32(1e-7))) / sigma_t

    var med = _medium(RGB(Float32(1.0), Float32(1.0), Float32(1.0)), RGB(Float32(1.5), Float32(1.5), Float32(1.5)))
    # sigma_t (red channel) = sigma_a.r + sigma_s.r = 1.0 + 1.5 = 2.5
    var pcg = PCG32(seed, seq)
    var t_surf = Float32(1.0e6)  # large enough that collision is essentially certain
    var ff = sample_homogeneous_free_flight(med, t_surf, pcg)
    assert_true(_close(ff.sig_t, sigma_t))
    assert_true(_close(ff.t_free, expected_t_free))
    assert_true(ff.collided == (expected_t_free < t_surf))

def test_free_flight_no_collision_gives_per_channel_beer_lambert_transmittance() raises:
    """When t_free >= t_surf (short segment / low extinction), the returned
    transmittance must be exp(-sigma_t_channel * t_surf) evaluated
    PER CHANNEL — not just the red/hero channel used to drive sampling."""
    var sigma_a = RGB(Float32(0.1), Float32(0.2), Float32(0.3))
    var sigma_s = RGB(Float32(0.05), Float32(0.05), Float32(0.05))
    var med = _medium(sigma_a, sigma_s)
    var t_surf = Float32(0.5)
    # Force "no collision" deterministically: pick a seed whose first draw
    # gives a t_free >= t_surf for this (small) sigma_t.r, by using a tiny
    # t_surf and checking the outcome is self-consistent either way.
    var pcg = PCG32(UInt64(77), UInt64(3))
    var ff = sample_homogeneous_free_flight(med, t_surf, pcg)
    if not ff.collided:
        var exp_r = exp(-(sigma_a.r + sigma_s.r) * t_surf)
        var exp_g = exp(-(sigma_a.g + sigma_s.g) * t_surf)
        var exp_b = exp(-(sigma_a.b + sigma_s.b) * t_surf)
        assert_true(_close(ff.transmittance.r, exp_r))
        assert_true(_close(ff.transmittance.g, exp_g))
        assert_true(_close(ff.transmittance.b, exp_b))
    else:
        # sigma_t.r == 0.15 with t_surf == 0.5 collided anyway: still a
        # valid outcome (t_free < t_surf), just doesn't exercise the branch
        # this test targets. Assert the collided-branch invariant instead
        # so the test isn't a no-op either way.
        assert_true(ff.t_free < t_surf)

# ── sample_area_light_uniform ────────────────────────────────────────────────

def _make_triangle_mesh(p0: SIMD[DType.float32, 3], p1: SIMD[DType.float32, 3], p2: SIMD[DType.float32, 3]) -> TriangleMesh_C:
    # Stride 4 floats/vertex: x,y,z,pad.
    var points = alloc[Float32](4 * 3)
    points[0*4+0] = p0[0]; points[0*4+1] = p0[1]; points[0*4+2] = p0[2]; points[0*4+3] = Float32(0.0)
    points[1*4+0] = p1[0]; points[1*4+1] = p1[1]; points[1*4+2] = p1[2]; points[1*4+3] = Float32(0.0)
    points[2*4+0] = p2[0]; points[2*4+1] = p2[1]; points[2*4+2] = p2[2]; points[2*4+3] = Float32(0.0)
    var vidx = alloc[Int64](3)
    vidx[0] = 0; vidx[1] = 1; vidx[2] = 2
    return TriangleMesh_C(
        points, UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(), vidx,
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    )

def test_sample_area_light_uniform_point_is_a_convex_combination_of_vertices() raises:
    """With a single light / single triangle, li and ti are forced to 0
    (n_lights=1, n_tris=1 both mod to 0), so the only randomness left is the
    barycentric (ru1,ru2) draw. Verify the returned point matches the exact
    closed-form barycentric formula the code uses, by peeking the same two
    draws (after the li/ti draws it also consumes) from an identically
    seeded generator."""
    var p0 = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var p1 = SIMD[DType.float32, 3](2.0, 0.0, 0.0)
    var p2 = SIMD[DType.float32, 3](0.0, 3.0, 0.0)
    var mesh = _make_triangle_mesh(p0, p1, p2)
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh

    var al = AreaLight_C(Int32(0), Int32(1), RGB(Float32(1.0)), Float32(3.0))
    var lights = alloc[AreaLight_C](1)
    lights[0] = al

    var seed = UInt64(4242)
    var seq = UInt64(21)
    var peek = PCG32(seed, seq)
    _ = peek.next_uint()  # li draw (n_lights=1 -> forced 0, but still consumed)
    _ = peek.next_uint()  # ti draw (n_tris=1 -> forced 0, but still consumed)
    var ru1 = peek.next_float()
    var ru2 = peek.next_float()
    var sr1 = sqrt(ru1)
    var expected = p0 * (Float32(1.0) - sr1) + p1 * (sr1 * (Float32(1.0) - ru2)) + p2 * (sr1 * ru2)

    var pcg = PCG32(seed, seq)
    var s = sample_area_light_uniform(lights, meshes, 1, pcg)
    assert_true(_simd_close(s.point, expected))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()
    lights.free()

def test_sample_area_light_uniform_normal_matches_geometric_normal_when_no_shading_normals() raises:
    """With no per-vertex normals on the light mesh (sentinel pointer), the
    returned normal must be exactly the normalized geometric triangle
    normal cross(p1-p0, p2-p0) — no shading-normal averaging applies."""
    var p0 = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var p1 = SIMD[DType.float32, 3](1.0, 0.0, 0.0)
    var p2 = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
    var mesh = _make_triangle_mesh(p0, p1, p2)
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh
    var al = AreaLight_C(Int32(0), Int32(1), RGB(Float32(1.0)), Float32(0.5))
    var lights = alloc[AreaLight_C](1)
    lights[0] = al

    var pcg = PCG32(UInt64(88), UInt64(2))
    var s = sample_area_light_uniform(lights, meshes, 1, pcg)
    assert_true(_simd_close(s.normal, SIMD[DType.float32, 3](0.0, 0.0, 1.0)))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()
    lights.free()

# ── _geom_normal / _shading_normal_at ────────────────────────────────────────

def _make_hit(u: Float32, v: Float32) -> Intersection_C:
    var pid = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    return Intersection_C(pid, Float32(1.0), u, v, Int8(1), Int8(0), Int8(0), Int8(0))

def test_geom_normal_matches_cross_product_of_the_triangle_edges() raises:
    """A right triangle at the origin: cross((1,0,0),(0,1,0)) = (0,0,1),
    already unit length — this is the flat per-triangle normal with no
    barycentric dependence at all."""
    var p0 = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var p1 = SIMD[DType.float32, 3](1.0, 0.0, 0.0)
    var p2 = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
    var mesh = _make_triangle_mesh(p0, p1, p2)
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh

    var inter = _make_hit(Float32(0.25), Float32(0.25))
    var n = _geom_normal(inter, meshes)
    assert_true(_simd_close(n, SIMD[DType.float32, 3](0.0, 0.0, 1.0)))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()

def test_shading_normal_at_falls_back_to_geometric_when_no_vertex_normals() raises:
    var p0 = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var p1 = SIMD[DType.float32, 3](1.0, 0.0, 0.0)
    var p2 = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
    var mesh = _make_triangle_mesh(p0, p1, p2)
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh

    var inter = _make_hit(Float32(0.3), Float32(0.3))
    var sn = _shading_normal_at(inter, meshes)
    assert_true(_simd_close(sn, SIMD[DType.float32, 3](0.0, 0.0, 1.0)))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()

def test_shading_normal_at_interpolates_per_vertex_normals_barycentrically() raises:
    """With per-vertex normals present, the shading normal must be the
    normalized barycentric blend n0*(1-u-v) + n1*u + n2*v (matching the
    intersection's own (u,v), which here are chosen so the flat geometric
    normal (0,0,1) and the blend agree in sign, so no flip is triggered)."""
    var p0 = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var p1 = SIMD[DType.float32, 3](1.0, 0.0, 0.0)
    var p2 = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
    var mesh = _make_triangle_mesh(p0, p1, p2)

    var n0 = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    var n1 = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    var n2 = SIMD[DType.float32, 3](0.267261, 0.534522, 0.801784)  # normalize(1,2,3), tilted but same hemisphere
    var normals = alloc[Float32](3 * 3)
    normals[0] = n0[0]; normals[1] = n0[1]; normals[2] = n0[2]
    normals[3] = n1[0]; normals[4] = n1[1]; normals[5] = n1[2]
    normals[6] = n2[0]; normals[7] = n2[1]; normals[8] = n2[2]
    mesh.normals = normals
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh

    var u = Float32(0.2); var v = Float32(0.3)
    var w0 = Float32(1.0) - u - v
    var expected = n0 * w0 + n1 * u + n2 * v
    var el = sqrt(dot(expected, expected))
    expected = expected * (Float32(1.0) / el)

    var inter = _make_hit(u, v)
    var sn = _shading_normal_at(inter, meshes)
    assert_true(_simd_close(sn, expected))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes[0].normals.free()
    meshes.free()

# ── _dielectric_bounce ───────────────────────────────────────────────────────

def test_dielectric_bounce_normal_incidence_output_is_always_colinear() raises:
    """At exactly normal incidence, sin2_t is always 0 (no bend possible),
    so BOTH branches — reflect and refract — must produce a direction
    exactly colinear with the incoming ray's axis (straight back if
    reflected, straight through if refracted). This holds regardless of
    which branch pcg's Fresnel draw happens to pick, so it is a genuine
    invariant of the function, not a tautology tied to one branch."""
    var ray_dir = SIMD[DType.float32, 3](0.0, 0.0, -1.0)
    var hit_point = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var geom_normal = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    var pcg = PCG32(UInt64(1), UInt64(1))
    var (new_dir, _) = _dielectric_bounce(ray_dir, hit_point, geom_normal, Float32(1.5), 0, pcg)
    assert_true(_close(_simd_len(new_dir), Float32(1.0)))
    var colinear = _close(abs(new_dir[2]), Float32(1.0)) and _close(new_dir[0], Float32(0.0)) and _close(new_dir[1], Float32(0.0))
    assert_true(colinear)

def test_dielectric_bounce_total_internal_reflection_obeys_reflection_law() raises:
    """A grazing ray exiting a denser medium (dot(ray,geom_normal) > 0, so
    bounce>0 keeps `entering=False`) must hit total internal reflection and
    take the mirror-reflection branch: refl = ray_dir - 2*dot(ray_dir,n)*n,
    i.e. reflect(-ray_dir, n) in the wo/n convention used by geometry.reflect
    (wo must point away from the surface, so it's the negated ray direction)."""
    var geom_normal = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    var ray_dir = Vec3f(0.99, 0.0, 0.1411).normalize().to_simd()  # exiting (dot(ray,n) > 0)
    var pcg = PCG32(UInt64(1), UInt64(1))
    var (new_dir, _) = _dielectric_bounce(ray_dir, SIMD[DType.float32, 3](0.0), geom_normal, Float32(1.5), 1, pcg)
    var wo = -Vec3f(ray_dir[0], ray_dir[1], ray_dir[2])
    var expected = reflect(wo, Vec3f(0.0, 0.0, 1.0)).to_simd()
    assert_true(_simd_close(new_dir, expected))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
