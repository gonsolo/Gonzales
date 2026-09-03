from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import RGB, INV_PI, Vec3f
from gonzales.reservoir import ReservoirState, reservoir_update
from gonzales.restir_gi import (
    GIReservoir, gi_reservoir_init, gi_target_pdf,
    GIReservoirIO, gi_reservoir_io_null, gi_temporal_spatial_combine,
)
from gonzales.rng import PCG32

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── gi_reservoir_init ────────────────────────────────────────────────────────

def test_gi_reservoir_init_has_no_winner() raises:
    var res = gi_reservoir_init()
    assert_true(res.valid == Int8(0))
    assert_true(res.recon_is_delta == Int8(0))
    assert_true(_close(res.state.w_sum, Float32(0.0)))
    assert_true(_close(res.state.m, Float32(0.0)))

# ── gi_target_pdf ────────────────────────────────────────────────────────────
# Same physical setup as test_restir_di.mojo's di_target_pdf tests, since
# the two target functions share the exact same G-term shape -- only the
# "light point/normal/Le" role is renamed to "reconnection point/normal/Lo".

def test_gi_target_pdf_recon_directly_above_matches_hand_computation() raises:
    """Flat surface at origin (normal +Y), reconnection vertex 2 units
    straight up with its normal facing straight down at x1 -- cos_x1=1,
    cos_x2=1 (checked for backface rejection only, not multiplied into the
    weight -- see gi_target_pdf's own docstring for why: `lo` is already
    outgoing radiance toward x1, not an area-measure emission needing the
    full G=cos_x1*cos_x2/dist^2 conversion). p_hat = luminance(alb * lo *
    (1/pi) * cos_x1)."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8), Float32(0.8), Float32(0.8))
    var recon_point = Vec3f(Float32(0), Float32(2), Float32(0))
    var recon_normal = Vec3f(Float32(0), Float32(-1), Float32(0))
    var lo = RGB(Float32(10.0), Float32(10.0), Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    var cos_x1 = Float32(1.0)  # recon_point is straight up from hit_point
    var expected_channel = Float32(0.8) * Float32(10.0) * INV_PI * cos_x1
    assert_true(_close(p_hat, expected_channel))

def test_gi_target_pdf_zero_when_recon_below_surface_horizon() raises:
    """Surface normal +Y, reconnection point BELOW the surface -- cos_x1 <=
    0, must return exactly 0."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_point = Vec3f(Float32(0), Float32(-2), Float32(0))
    var recon_normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var lo = RGB(Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    assert_true(_close(p_hat, Float32(0.0)))

def test_gi_target_pdf_zero_when_recon_faces_away() raises:
    """Reconnection point geometrically above x1, but its OWN normal points
    further away (toward x1's side) -- cos_x2 <= 0, its back face is turned
    toward x1."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_point = Vec3f(Float32(0), Float32(2), Float32(0))
    var recon_normal = Vec3f(Float32(0), Float32(1), Float32(0))  # facing up, away from x1
    var lo = RGB(Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    assert_true(_close(p_hat, Float32(0.0)))

def test_gi_target_pdf_zero_at_degenerate_distance() raises:
    var hit_point = Vec3f(Float32(1), Float32(1), Float32(1))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_point = Vec3f(Float32(1), Float32(1), Float32(1))
    var recon_normal = Vec3f(Float32(0), Float32(-1), Float32(0))
    var lo = RGB(Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    assert_true(_close(p_hat, Float32(0.0)))

def test_gi_target_pdf_independent_of_distance_at_fixed_angles() raises:
    """Deliberately NOT inverse-square falloff, unlike di_target_pdf: `lo`
    is already outgoing radiance toward x1 (radiance is invariant along a
    ray), so with cos_x1/cos_x2 held fixed (straight up in both cases),
    p_hat must be IDENTICAL regardless of distance -- this is the exact
    property whose absence was a real, shipped bug (see gi_target_pdf's
    docstring): including a spurious 1/dist^2 term caused state.w to
    explode whenever spatial/temporal reuse combined candidates at
    different distances, since gi_resolve's own contribution never had
    that distance dependence to begin with."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_normal = Vec3f(Float32(0), Float32(-1), Float32(0))
    var lo = RGB(Float32(10.0))
    var far = gi_target_pdf(hit_point, normal, alb, Vec3f(Float32(0), Float32(4), Float32(0)), recon_normal, lo)
    var near = gi_target_pdf(hit_point, normal, alb, Vec3f(Float32(0), Float32(2), Float32(0)), recon_normal, lo)
    assert_true(_close(near, far))

# ── gi_temporal_spatial_combine ──────────────────────────────────────────────
# ctx-free by design (see restir_gi.mojo's module header), so unlike
# shading.mojo's di_temporal_step this is directly unit-testable with
# synthetic reservoirs -- no rendering involved. Tests focus on invariants
# that hold regardless of the PCG stream (reservoir_combine's m-accumulation
# is unconditional; delta/invalid neighbors must never be folded in at all)
# rather than exact RIS-acceptance outcomes, which are legitimately
# probabilistic.

def _make_valid(recon_point: Vec3f, recon_normal: Vec3f, lo: RGB, w: Float32, m: Float32, is_delta: Int8 = Int8(0)) -> GIReservoir:
    var res = gi_reservoir_init()
    res.recon_point = recon_point
    res.recon_normal = recon_normal
    res.lo = lo
    res.valid = Int8(1)
    res.recon_is_delta = is_delta
    res.state.w = w
    res.state.m = m
    return res

def test_gi_combine_no_temporal_finalizes_plain_single_candidate() raises:
    """`pixel_idx < 0` (batch/no-history mode): must behave as a plain
    single-candidate RIS finalize -- no crash touching gi_reservoir_io_null()'s
    dangling pointers, m unchanged, W = w_sum / (m * p_hat)."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_point = Vec3f(Float32(0), Float32(2), Float32(0))
    var recon_normal = Vec3f(Float32(0), Float32(-1), Float32(0))
    var lo = RGB(Float32(10.0))

    var res = gi_reservoir_init()
    _ = reservoir_update(res.state, Float32(5.0), Float32(0.0))
    res.recon_point = recon_point
    res.recon_normal = recon_normal
    res.lo = lo
    res.valid = Int8(1)

    var pcg = PCG32(UInt64(1), UInt64(1))
    gi_temporal_spatial_combine(res, hit_point, normal, alb, pcg, gi_reservoir_io_null(), pixel_idx=-1)

    var expected_p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    assert_true(_close(res.state.m, Float32(1.0)))
    assert_true(_close(res.state.w, Float32(5.0) / expected_p_hat))

def test_gi_combine_temporal_accumulates_confidence_regardless_of_winner() raises:
    """`reservoir_combine`'s m += src.m is unconditional (reservoir.mojo's own
    documented contract) -- a valid, non-delta previous-frame reservoir must
    add its full m to this pixel's, whether or not it ends up winning."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var fresh = _make_valid(
        Vec3f(Float32(0), Float32(2), Float32(0)),
        Vec3f(Float32(0), Float32(-1), Float32(0)),
        RGB(Float32(10.0)), Float32(5.0), Float32(1.0))
    var prev = _make_valid(
        Vec3f(Float32(0), Float32(3), Float32(0)),
        Vec3f(Float32(0), Float32(-1), Float32(0)),
        RGB(Float32(20.0)), Float32(2.0), Float32(3.0))

    var prev_buf = List[GIReservoir]()
    prev_buf.append(prev)
    var write_buf = List[GIReservoir]()
    write_buf.append(gi_reservoir_init())
    var io = GIReservoirIO(
        read=prev_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](), write=write_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),  # spatial disabled -- isolates temporal-only behavior
    )
    var pcg = PCG32(UInt64(12345), UInt64(1))
    var res = fresh
    gi_temporal_spatial_combine(res, hit_point, normal, alb, pcg, io, pixel_idx=0)
    _ = prev_buf^; _ = write_buf^
    assert_true(_close(res.state.m, Float32(1.0) + Float32(3.0)))

def test_gi_combine_rejects_delta_flagged_previous_reservoir_entirely() raises:
    """Phase 4.3: a reconnection vertex flagged recon_is_delta must never be
    folded in at all (not just never win) -- m must stay exactly this
    pixel's own fresh m, as if no temporal history existed."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var fresh = _make_valid(
        Vec3f(Float32(0), Float32(2), Float32(0)),
        Vec3f(Float32(0), Float32(-1), Float32(0)),
        RGB(Float32(10.0)), Float32(5.0), Float32(1.0))
    var prev = _make_valid(
        Vec3f(Float32(0), Float32(3), Float32(0)),
        Vec3f(Float32(0), Float32(-1), Float32(0)),
        RGB(Float32(20.0)), Float32(2.0), Float32(3.0), is_delta=Int8(1))

    var prev_buf = List[GIReservoir]()
    prev_buf.append(prev)
    var write_buf = List[GIReservoir]()
    write_buf.append(gi_reservoir_init())
    var io = GIReservoirIO(
        read=prev_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](), write=write_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),
    )
    var pcg = PCG32(UInt64(12345), UInt64(1))
    var res = fresh
    gi_temporal_spatial_combine(res, hit_point, normal, alb, pcg, io, pixel_idx=0)
    _ = prev_buf^; _ = write_buf^
    assert_true(_close(res.state.m, Float32(1.0)))

def test_gi_combine_rejects_invalid_previous_reservoir() raises:
    """A previous-frame slot with no real candidate yet (valid=0, e.g. a
    pixel just uncovered by camera motion) must be excluded the same way,
    not treated as a real zero-weight candidate."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var fresh = _make_valid(
        Vec3f(Float32(0), Float32(2), Float32(0)),
        Vec3f(Float32(0), Float32(-1), Float32(0)),
        RGB(Float32(10.0)), Float32(5.0), Float32(1.0))
    var prev = gi_reservoir_init()  # valid == 0
    prev.state.m = Float32(3.0)     # if this leaked in despite valid==0, m would jump

    var prev_buf = List[GIReservoir]()
    prev_buf.append(prev)
    var write_buf = List[GIReservoir]()
    write_buf.append(gi_reservoir_init())
    var io = GIReservoirIO(
        read=prev_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](), write=write_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),
    )
    var pcg = PCG32(UInt64(12345), UInt64(1))
    var res = fresh
    gi_temporal_spatial_combine(res, hit_point, normal, alb, pcg, io, pixel_idx=0)
    _ = prev_buf^; _ = write_buf^
    assert_true(_close(res.state.m, Float32(1.0)))

def test_gi_combine_zero_target_pdf_at_winner_gives_zero_weight() raises:
    """`reservoir_finalize`'s documented degenerate case: a winning candidate
    whose target pdf evaluates to 0 AT THIS PIXEL (here: the reconnection
    point sits below x1's horizon) must finalize to W=0, not NaN/Inf."""
    var hit_point = Vec3f(Float32(0), Float32(0), Float32(0))
    var normal = Vec3f(Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var res = _make_valid(
        Vec3f(Float32(0), Float32(-2), Float32(0)),  # below horizon
        Vec3f(Float32(0), Float32(1), Float32(0)),
        RGB(Float32(10.0)), Float32(5.0), Float32(1.0))
    var pcg = PCG32(UInt64(1), UInt64(1))
    gi_temporal_spatial_combine(res, hit_point, normal, alb, pcg, gi_reservoir_io_null(), pixel_idx=-1)
    assert_true(_close(res.state.w, Float32(0.0)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
