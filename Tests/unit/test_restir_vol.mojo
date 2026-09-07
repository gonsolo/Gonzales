# Volumetric ReSTIR (Phase 7) payload + target-function + combine math.
# Pure unit tests: no rendering, no medium sampler, synthetic reservoirs only --
# the same scope test_restir_gi.mojo covers for GI, and possible for the same
# reason (restir_vol.mojo deliberately holds no dependency on its caller).

from std.math import abs, sqrt
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import RGB, hg_phase, Vec3f
from gonzales.reservoir import ReservoirState, reservoir_update
from gonzales.restir_vol import (
    VolReservoir, vol_reservoir_init, vol_target_pdf, vol_shift_scatter_vertex,
    VolShiftMode, VolReservoirIO, vol_reservoir_io_null,
    vol_temporal_spatial_combine, VOL_TR_UNIT, VOL_MAX_FINALIZED_WEIGHT,
)
from gonzales.rng import PCG32

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _rel_close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) <= Float32(1e-3) * max(abs(a), abs(b)) + EPS

# ── vol_target_pdf ───────────────────────────────────────────────────────────

def test_vol_target_pdf_matches_hand_computation() raises:
    """Straight-line geometry with every factor known: a vertex at the origin,
    a light 2 units above facing straight down, camera ray travelling +z. The
    target must be exactly luminance(Le) x sigma_s x hg_phase x cos_l/dist^2."""
    var ray_dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var light_p = Vec3f(Float32(0), Float32(2), Float32(0))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))  # faces the vertex
    var le = RGB(Float32(3.0), Float32(3.0), Float32(3.0))
    var sigma_s = Float32(0.5)
    var g = Float32(0.0)

    var got = vol_target_pdf(ray_dir, scatter, sigma_s, g, light_p, light_n, le, VOL_TR_UNIT)

    # wi = +y, wo = -ray_dir = -z, so dot(wo, wi) = 0.
    var ph = hg_phase(Float32(0.0), g)
    var scale = sigma_s * ph * Float32(1.0) * (Float32(1.0) / Float32(4.0))  # cos_l=1, dist^2=4
    var expected = Float32(3.0) * scale  # luminance of a grey RGB is the grey
    assert_true(_rel_close(got, expected))

def test_vol_target_pdf_scales_linearly_with_the_transmittance_seam() raises:
    """`tr` is a plain multiplicative seam (this is what lets a caller pass
    VOL_TR_UNIT to get the transmittance-free target the newer formulation
    wants). Halving it must halve the target exactly."""
    var ray_dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var light_p = Vec3f(Float32(0), Float32(2), Float32(0))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var le = RGB(Float32(3.0))
    var full = vol_target_pdf(ray_dir, scatter, Float32(0.5), Float32(0), light_p, light_n, le, VOL_TR_UNIT)
    var half = vol_target_pdf(ray_dir, scatter, Float32(0.5), Float32(0), light_p, light_n, le, Float32(0.5))
    assert_true(_rel_close(half, full * Float32(0.5)))

def test_vol_target_pdf_zero_when_light_faces_away() raises:
    """Emission leaves the front face only -- a light whose normal points away
    from the vertex must score exactly 0, not a small positive."""
    var ray_dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var light_p = Vec3f(Float32(0), Float32(2), Float32(0))
    var away = Vec3f(Float32(0), Float32(1), Float32(0))  # points AWAY from the vertex
    var got = vol_target_pdf(ray_dir, scatter, Float32(0.5), Float32(0), light_p, away, RGB(Float32(3.0)), VOL_TR_UNIT)
    assert_true(_close(got, Float32(0.0)))

def test_vol_target_pdf_zero_for_vanishing_sigma_s_or_transmittance() raises:
    """No scattering coefficient means no scattering event to resample; zero
    transmittance means nothing reaches the vertex. Both are hard zeros."""
    var ray_dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var light_p = Vec3f(Float32(0), Float32(2), Float32(0))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var le = RGB(Float32(3.0))
    assert_true(_close(vol_target_pdf(ray_dir, scatter, Float32(0.0), Float32(0), light_p, light_n, le, VOL_TR_UNIT), Float32(0.0)))
    assert_true(_close(vol_target_pdf(ray_dir, scatter, Float32(0.5), Float32(0), light_p, light_n, le, Float32(0.0)), Float32(0.0)))

def test_vol_target_pdf_zero_at_degenerate_distance() raises:
    """A light sample coincident with the scattering vertex has no direction
    and an infinite 1/dist^2 -- must be rejected, not returned as inf/NaN."""
    var ray_dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var p = Vec3f(Float32(1), Float32(2), Float32(3))
    var got = vol_target_pdf(ray_dir, p, Float32(0.5), Float32(0), p,
                             Vec3f(Float32(0), Float32(-1), Float32(0)), RGB(Float32(3.0)), VOL_TR_UNIT)
    assert_true(_close(got, Float32(0.0)))

def test_vol_target_pdf_has_no_cosine_at_the_scattering_vertex() raises:
    """A phase function is not cosine-weighted, unlike a surface BSDF. Pinned
    by construction: with g=0 the phase value is constant over direction, so
    rotating the LIGHT around the vertex at fixed distance must leave the
    target completely unchanged. A stray cos at the vertex would make it vary.
    (This is the volumetric analogue of the geometry-term bug gi_target_pdf's
    docstring records.)"""
    var ray_dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var le = RGB(Float32(3.0))
    var d = Float32(2.0)

    # Light directly above, facing down.
    var above = vol_target_pdf(ray_dir, scatter, Float32(0.5), Float32(0),
                               Vec3f(Float32(0), d, Float32(0)),
                               Vec3f(Float32(0), Float32(-1), Float32(0)), le, VOL_TR_UNIT)
    # Same distance, off to the side, still facing the vertex.
    var side = vol_target_pdf(ray_dir, scatter, Float32(0.5), Float32(0),
                              Vec3f(d, Float32(0), Float32(0)),
                              Vec3f(Float32(-1), Float32(0), Float32(0)), le, VOL_TR_UNIT)
    assert_true(_rel_close(above, side))

def test_vol_target_pdf_forward_scattering_favours_the_light_ahead() raises:
    """Pins the wo/wi convention, which is exactly the kind of sign that breaks
    silently. hg_phase takes dot(wo, wi) with wo pointing BACK along the ray,
    and g > 0 peaks at dot = -1, i.e. wi CONTINUING in the ray's direction of
    travel. So for a forward-scattering medium a light further along the ray
    must score higher than one behind, at equal distance."""
    var ray_dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var le = RGB(Float32(3.0))
    var g = Float32(0.8)
    var d = Float32(2.0)

    var ahead = vol_target_pdf(ray_dir, scatter, Float32(0.5), g,
                               Vec3f(Float32(0), Float32(0), d),
                               Vec3f(Float32(0), Float32(0), Float32(-1)), le, VOL_TR_UNIT)
    var behind = vol_target_pdf(ray_dir, scatter, Float32(0.5), g,
                                Vec3f(Float32(0), Float32(0), -d),
                                Vec3f(Float32(0), Float32(0), Float32(1)), le, VOL_TR_UNIT)
    assert_true(ahead > behind)

# ── vol_shift_scatter_vertex (seam 2) ────────────────────────────────────────

def test_vol_shift_identity_reuses_the_world_space_vertex_verbatim() raises:
    var p = Vec3f(Float32(1), Float32(2), Float32(3))
    var (ok, got) = vol_shift_scatter_vertex(
        VolShiftMode.identity, p, Vec3f(Float32(9), Float32(9), Float32(9)),
        Vec3f(Float32(0), Float32(0), Float32(1)))
    assert_true(ok)
    assert_true(_close(got.x, p.x) and _close(got.y, p.y) and _close(got.z, p.z))

def test_vol_shift_ghost_mode_is_refused_not_silently_aliased() raises:
    """The ghost-vertex bijection has no published derivation yet. Asking for
    it must FAIL rather than quietly return the identity map's answer, or a
    caller would believe it had the newer formulation while getting the older
    one."""
    var (ok, _got) = vol_shift_scatter_vertex(
        VolShiftMode.ghost, Vec3f(Float32(1), Float32(2), Float32(3)),
        Vec3f(Float32(0)), Vec3f(Float32(0), Float32(0), Float32(1)))
    assert_false(ok)

# ── vol_temporal_spatial_combine ─────────────────────────────────────────────

def _make_valid(scatter: Vec3f, light_p: Vec3f, light_n: Vec3f, le: RGB,
                w: Float32, m: Float32, medium_idx: Int32 = Int32(0)) -> VolReservoir:
    var res = vol_reservoir_init()
    res.scatter_point = scatter
    res.light_point = light_p
    res.light_normal = light_n
    res.le = le
    res.sigma_s = Float32(0.5)
    res.phase_g = Float32(0.0)
    res.medium_idx = medium_idx
    res.valid = Int8(1)
    res.state.w = w
    res.state.m = m
    return res

def test_vol_combine_no_temporal_finalizes_plain_single_candidate() raises:
    """`pixel_idx < 0` (batch/no-history mode): a plain single-candidate RIS
    finalize, with no crash touching vol_reservoir_io_null()'s dangling
    pointers. m unchanged, W = w_sum / (m * p_hat)."""
    var ray_o = Vec3f(Float32(0), Float32(0), Float32(-1))
    var ray_d = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var light_p = Vec3f(Float32(0), Float32(2), Float32(0))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var le = RGB(Float32(3.0))

    var res = vol_reservoir_init()
    _ = reservoir_update(res.state, Float32(5.0), Float32(0.0))
    res.scatter_point = scatter
    res.light_point = light_p
    res.light_normal = light_n
    res.le = le
    res.sigma_s = Float32(0.5)
    res.phase_g = Float32(0.0)
    res.medium_idx = Int32(0)
    res.valid = Int8(1)

    var pcg = PCG32(UInt64(1), UInt64(1))
    vol_temporal_spatial_combine(res, ray_o, ray_d, Int32(0), pcg,
                                 vol_reservoir_io_null(), pixel_idx=-1)

    var expected_p_hat = vol_target_pdf(ray_d, scatter, Float32(0.5), Float32(0),
                                        light_p, light_n, le, VOL_TR_UNIT)
    assert_true(_close(res.state.m, Float32(1.0)))
    assert_true(_rel_close(res.state.w, Float32(5.0) / expected_p_hat))

def test_vol_combine_does_not_clamp_a_legitimately_large_weight() raises:
    """For a single candidate W = w_sum / (m * p_hat) is exactly 1/q, the
    reciprocal of the source sampling pdf -- unbounded by construction, and
    genuinely large whenever distance sampling picked an unlikely vertex. This
    geometry produces W ~= 168 from entirely healthy inputs, which is why
    VOL_MAX_FINALIZED_WEIGHT does not carry GI's measured threshold of 10:
    that would have silently cost this case a factor of 16. Guards against
    someone "restoring consistency" with the other two reservoirs by copying
    the constant across."""
    var ray_o = Vec3f(Float32(0), Float32(0), Float32(-1))
    var ray_d = Vec3f(Float32(0), Float32(0), Float32(1))
    var scatter = Vec3f(Float32(0), Float32(0), Float32(0))
    var light_p = Vec3f(Float32(0), Float32(2), Float32(0))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var le = RGB(Float32(3.0))

    var res = vol_reservoir_init()
    _ = reservoir_update(res.state, Float32(5.0), Float32(0.0))
    res.scatter_point = scatter
    res.light_point = light_p
    res.light_normal = light_n
    res.le = le
    res.sigma_s = Float32(0.5)
    res.phase_g = Float32(0.0)
    res.medium_idx = Int32(0)
    res.valid = Int8(1)

    var pcg = PCG32(UInt64(1), UInt64(1))
    vol_temporal_spatial_combine(res, ray_o, ray_d, Int32(0), pcg,
                                 vol_reservoir_io_null(), pixel_idx=-1)
    assert_true(res.state.w > Float32(100.0))
    assert_true(res.state.w < VOL_MAX_FINALIZED_WEIGHT)

def _io_temporal_only(mut prev_buf: List[VolReservoir], mut write_buf: List[VolReservoir]) -> VolReservoirIO:
    # frame_w/frame_h = 0 disables the spatial pass, isolating temporal behaviour.
    return VolReservoirIO(
        read=prev_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        write=write_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),
    )

def test_vol_combine_temporal_accumulates_confidence_regardless_of_winner() raises:
    """reservoir_combine's `m += src.m` is unconditional (reservoir.mojo's
    documented contract): a compatible previous-frame reservoir adds its full
    m whether or not it wins."""
    var ray_o = Vec3f(Float32(0), Float32(0), Float32(-1))
    var ray_d = Vec3f(Float32(0), Float32(0), Float32(1))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var fresh = _make_valid(Vec3f(Float32(0)), Vec3f(Float32(0), Float32(2), Float32(0)),
                            light_n, RGB(Float32(3.0)), Float32(5.0), Float32(1.0))
    var prev = _make_valid(Vec3f(Float32(0), Float32(0), Float32(0.5)),
                           Vec3f(Float32(0), Float32(3), Float32(0)),
                           light_n, RGB(Float32(6.0)), Float32(2.0), Float32(3.0))

    var prev_buf = List[VolReservoir]()
    prev_buf.append(prev)
    var write_buf = List[VolReservoir]()
    write_buf.append(vol_reservoir_init())
    var io = _io_temporal_only(prev_buf, write_buf)

    var pcg = PCG32(UInt64(12345), UInt64(1))
    var res = fresh
    vol_temporal_spatial_combine(res, ray_o, ray_d, Int32(0), pcg, io, pixel_idx=0)
    _ = prev_buf^; _ = write_buf^
    assert_true(_close(res.state.m, Float32(1.0) + Float32(3.0)))

def test_vol_combine_rejects_previous_reservoir_from_a_different_medium() raises:
    """A vertex in another medium is not a low-quality candidate, it is a
    meaningless one -- it must not even contribute confidence."""
    var ray_o = Vec3f(Float32(0), Float32(0), Float32(-1))
    var ray_d = Vec3f(Float32(0), Float32(0), Float32(1))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var fresh = _make_valid(Vec3f(Float32(0)), Vec3f(Float32(0), Float32(2), Float32(0)),
                            light_n, RGB(Float32(3.0)), Float32(5.0), Float32(1.0), Int32(0))
    var prev = _make_valid(Vec3f(Float32(0), Float32(0), Float32(0.5)),
                           Vec3f(Float32(0), Float32(3), Float32(0)),
                           light_n, RGB(Float32(6.0)), Float32(2.0), Float32(3.0), Int32(1))

    var prev_buf = List[VolReservoir]()
    prev_buf.append(prev)
    var write_buf = List[VolReservoir]()
    write_buf.append(vol_reservoir_init())
    var io = _io_temporal_only(prev_buf, write_buf)

    var pcg = PCG32(UInt64(7), UInt64(1))
    var res = fresh
    vol_temporal_spatial_combine(res, ray_o, ray_d, Int32(0), pcg, io, pixel_idx=0)
    _ = prev_buf^; _ = write_buf^
    assert_true(_close(res.state.m, Float32(1.0)))

def test_vol_combine_rejects_invalid_previous_reservoir() raises:
    var ray_o = Vec3f(Float32(0), Float32(0), Float32(-1))
    var ray_d = Vec3f(Float32(0), Float32(0), Float32(1))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var fresh = _make_valid(Vec3f(Float32(0)), Vec3f(Float32(0), Float32(2), Float32(0)),
                            light_n, RGB(Float32(3.0)), Float32(5.0), Float32(1.0))
    var prev = vol_reservoir_init()   # valid == 0
    prev.state.m = Float32(9.0)

    var prev_buf = List[VolReservoir]()
    prev_buf.append(prev)
    var write_buf = List[VolReservoir]()
    write_buf.append(vol_reservoir_init())
    var io = _io_temporal_only(prev_buf, write_buf)

    var pcg = PCG32(UInt64(3), UInt64(1))
    var res = fresh
    vol_temporal_spatial_combine(res, ray_o, ray_d, Int32(0), pcg, io, pixel_idx=0)
    _ = prev_buf^; _ = write_buf^
    assert_true(_close(res.state.m, Float32(1.0)))

def test_vol_combine_zero_target_pdf_at_winner_gives_zero_weight() raises:
    """A winner the target scores at 0 (here: the light faces away) must
    finalize to W = 0 rather than dividing by zero."""
    var ray_o = Vec3f(Float32(0), Float32(0), Float32(-1))
    var ray_d = Vec3f(Float32(0), Float32(0), Float32(1))
    var res = _make_valid(Vec3f(Float32(0)), Vec3f(Float32(0), Float32(2), Float32(0)),
                          Vec3f(Float32(0), Float32(1), Float32(0)),  # faces away
                          RGB(Float32(3.0)), Float32(5.0), Float32(1.0))
    var pcg = PCG32(UInt64(1), UInt64(1))
    vol_temporal_spatial_combine(res, ray_o, ray_d, Int32(0), pcg,
                                 vol_reservoir_io_null(), pixel_idx=-1)
    assert_true(_close(res.state.w, Float32(0.0)))

def test_vol_combine_persists_the_result_for_the_next_frame() raises:
    """With a real IO the finalized reservoir must land in `write`, which is
    what makes temporal reuse possible at all on the following frame."""
    var ray_o = Vec3f(Float32(0), Float32(0), Float32(-1))
    var ray_d = Vec3f(Float32(0), Float32(0), Float32(1))
    var light_n = Vec3f(Float32(0), Float32(-1), Float32(0))
    var fresh = _make_valid(Vec3f(Float32(0)), Vec3f(Float32(0), Float32(2), Float32(0)),
                            light_n, RGB(Float32(3.0)), Float32(5.0), Float32(1.0))
    var prev_buf = List[VolReservoir]()
    prev_buf.append(vol_reservoir_init())
    var write_buf = List[VolReservoir]()
    write_buf.append(vol_reservoir_init())
    var io = _io_temporal_only(prev_buf, write_buf)

    var pcg = PCG32(UInt64(11), UInt64(1))
    var res = fresh
    vol_temporal_spatial_combine(res, ray_o, ray_d, Int32(0), pcg, io, pixel_idx=0)
    var stored = write_buf[0]
    _ = prev_buf^; _ = write_buf^
    assert_true(stored.valid != Int8(0))
    assert_true(_close(stored.state.w, res.state.w))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
