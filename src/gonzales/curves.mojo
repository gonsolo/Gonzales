"""Native curve (hair/fur) primitive, split out of geometry.mojo (the
per-cluster module split -- see project_geometry_module_split memory). Only
depends on geometry.mojo's core (Point3f/Vec3f/cross/dot), confirmed by a
symbol-reference scan of the code with comments and docstrings stripped
before the split: geometry.mojo had zero real cross-cluster dependencies
except one (PathState_C holding a Ray_C), so this and its five siblings
(primitives.mojo, materials.mojo, lights.mojo, media.mojo, render_state.mojo)
extract cleanly with no cycles."""
from std.math import sqrt, min, max, abs
from .geometry import Point3f, Vec3f, cross, dot


# ── Native curve (hair/fur) primitive ──────────────────────────────────────────
# A single local cubic B-spline segment: 4 control points already windowed to
# this piece (mirrors PBRT's `Shape "curve"` per-segment convention). Stored
# and BVH-indexed natively — no triangle tessellation, no per-strand vertex
# buffers. Intersected lazily per ray as a short chain of locally-linear
# fixed-radius cylinders (see intersect_curve below).

@fieldwise_init
struct Curve_C(TrivialRegisterPassable):
    """One cubic B-spline curve segment. width0/width1 are the curve's full
    diameter at the segment's start/end (u=0/u=1 of THIS segment's local
    parameter range, not the whole original strand). n_pieces is a per-curve
    tessellation-at-trace-time hint (1..CURVE_N_PIECES), precomputed once at
    parse time from the control polygon's flatness — most real hair/fur
    exports chop each strand into thousands of already-near-straight
    directives, so forcing the full CURVE_N_PIECES chain on every one of
    them wastes GPU work and — because ray-adjacent pixels can pick very
    different piece counts — causes severe SIMT warp divergence."""
    var cp0: Point3f
    var cp1: Point3f
    var cp2: Point3f
    var cp3: Point3f
    var width0: Float32
    var width1: Float32
    var materialIndex: Int32
    var n_pieces: Int32

comptime CURVE_N_PIECES: Int = 7   # max locally-linear pieces per segment (8 samples)

# Divergence experiment (traverse_paths_gpu): max curve-leaf candidates recorded
# per ray before falling back to immediate intersection. See traverse_bvh2_core_defer_curves.
# The CUDA path's fallback (intersect_curve inline once over K) makes
# correctness independent of K there, so this only sizes the small,
# always-allocated curve_cand_prim_buf/curve_cand_offset_buf CUDA fallback
# buffers (gpu.mojo) -- kept modest since Vulkan-RT rendering doesn't use
# them at all anymore (see vulkaninterop_rt_create_scene's docstring): its
# own candidate pool is a SEPARATE, independently-sized Vulkan-interop
# buffer read directly by resolve_curve_candidates_gpu, with no CUDA-side
# duplicate to keep in sync -- see InteropRtScene::curveCandBuffer's
# comment (vulkaninterop.cpp) for that pool's own capacity constant.
comptime CURVE_DEFER_K: Int = 4

@always_inline
def _curve_perp_axis(t: Vec3f) -> Vec3f:
    """Deterministic unit vector perpendicular to t, built from t alone (no
    stored per-vertex data needed — reproducible at both intersect and shade time)."""
    var ux: Float32; var uy: Float32; var uz: Float32
    if abs(t[1]) < Float32(0.9):
        ux = -t[2]; uy = Float32(0.0); uz = t[0]
    else:
        ux = Float32(0.0); uy = t[2]; uz = -t[1]
    var v = Vec3f(ux, uy, uz)
    var l = sqrt(dot(v, v))
    if l > Float32(1e-12):
        return v * (Float32(1.0) / l)
    return Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))

@always_inline
def curve_bspline_point(curve: Curve_C, t: Float32) -> Vec3f:
    """Evaluate the uniform cubic B-spline segment at local parameter t in [0,1]."""
    var t2 = t * t
    var t3 = t2 * t
    var b0 = (Float32(1.0) - Float32(3.0)*t + Float32(3.0)*t2 - t3) / Float32(6.0)
    var b1 = (Float32(4.0) - Float32(6.0)*t2 + Float32(3.0)*t3) / Float32(6.0)
    var b2 = (Float32(1.0) + Float32(3.0)*t + Float32(3.0)*t2 - Float32(3.0)*t3) / Float32(6.0)
    var b3 = t3 / Float32(6.0)
    var p0 = curve.cp0.to_simd(); var p1 = curve.cp1.to_simd()
    var p2 = curve.cp2.to_simd(); var p3 = curve.cp3.to_simd()
    return p0*b0 + p1*b1 + p2*b2 + p3*b3

@always_inline
def curve_piece_endpoints(curve: Curve_C, piece: Int) -> Tuple[Vec3f, Vec3f, Float32, Float32]:
    """Sample points + radii bounding one of this curve's n_pieces locally-linear pieces."""
    var n = Float32(curve.n_pieces)
    var t0 = Float32(piece) / n
    var t1 = Float32(piece + 1) / n
    var q0 = curve_bspline_point(curve, t0)
    var q1 = curve_bspline_point(curve, t1)
    var r0 = (curve.width0 + (curve.width1 - curve.width0) * t0) * Float32(0.5)
    var r1 = (curve.width0 + (curve.width1 - curve.width0) * t1) * Float32(0.5)
    return (q0, q1, r0, r1)

@always_inline
def intersect_curve(
    ray_org: Vec3f,
    ray_dir: Vec3f,
    curve: Curve_C,
    first_piece: Int,
    piece_count: Int,
    tMax: Float32,
) -> Tuple[Bool, Float32, Float32, Float32]:
    """Ray vs. a small contiguous run of locally-linear pieces (fixed-radius
    cylinders) of a native curve segment: [first_piece, first_piece+piece_count).
    Returns (hit, t, h, v):
      h — signed fiber cross-section position in [-1,1] (PBRT HairBxDF convention)
      v — position along this curve segment's own [0,1] parameter range

    A BVH leaf covers one such run — see pbrt_parser.mojo's greedy
    piece-merging (adjacent pieces of the same curly curve get grouped into
    one leaf when they're still collinear enough that the group's bound
    isn't much looser than an individual piece's). piece_count is almost
    always 1-3 in practice (curls are locally gradual, not zigzag), only
    reaching curve.n_pieces in the rare case a whole curly segment is one
    flat run. This is what keeps the leaf count — and BVH build time — down
    without going back to one loose bound for the whole segment.

    Only the near root of the ray/infinite-cylinder quadratic is tested (not
    both roots) — deliberately simpler/leaner than a textbook capsule test.
    Adjacent pieces overlap at their shared endpoint, so a hit missed at one
    piece's far cap is picked up by the next piece's near-root test; the
    gap this could theoretically miss is far below pixel-visible scale given
    how short each piece already is.

    Measured @no_inline on an earlier version of this function: register
    count dropped only marginally and per-kernel GPU time got WORSE —
    call/return overhead outweighed any register win, so this stays
    @always_inline. Left as a note so this isn't retried blindly.
    """
    var n_pieces_f = Float32(curve.n_pieces)
    var best_hit = False
    var best_t = tMax
    var best_h = Float32(0.0)
    var best_v = Float32(0.0)

    for piece in range(first_piece, first_piece + piece_count):
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var r = (r0 + r1) * Float32(0.5)
        if r < Float32(1e-8):
            continue
        var axis = q1 - q0
        var axis_len_sq = dot(axis, axis)
        if axis_len_sq < Float32(1e-16):
            continue

        # Ray-vs-infinite-cylinder-around-`axis` via cross products — avoids
        # normalizing axis (no sqrt) for the common case where this piece
        # isn't hit at all; sqrt is only paid once a valid closer root exists.
        var oc = ray_org - q0
        var A = cross(axis, ray_dir)
        var B = cross(axis, oc)
        var a = dot(A, A)
        if a < Float32(1e-12):
            continue
        var b = Float32(2.0) * dot(A, B)
        var c = dot(B, B) - r*r*axis_len_sq
        var disc = b*b - Float32(4.0)*a*c
        if disc < Float32(0.0):
            continue

        var tc = (-b - sqrt(disc)) / (Float32(2.0) * a)
        if tc <= Float32(1e-4) or tc >= best_t:
            continue

        var axis_len = sqrt(axis_len_sq)
        var axis_dir = axis * (Float32(1.0) / axis_len)
        var hit_pos = (ray_org + tc*ray_dir) - q0
        var s = dot(hit_pos, axis_dir)
        if s < Float32(0.0) or s > axis_len:
            continue

        var offset_perp = hit_pos - s * axis_dir
        var u_axis = _curve_perp_axis(axis_dir)
        best_hit = True; best_t = tc
        best_h = max(Float32(-1.0), min(Float32(1.0), dot(offset_perp, u_axis) / r))
        best_v = (Float32(piece) + s/axis_len) / n_pieces_f

    return (best_hit, best_t, best_h, best_v)

@always_inline
def curve_piece_bounds(curve: Curve_C, piece: Int) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32]:
    """Tight AABB for one locally-linear piece: the segment q0..q1 thickened
    by max(r0,r1). Used as the BVH leaf bound now that leaves are per-piece
    instead of per-whole-curve (see intersect_curve)."""
    var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
    var r = max(r0, r1)
    var xmin = min(q0[0], q1[0]) - r
    var ymin = min(q0[1], q1[1]) - r
    var zmin = min(q0[2], q1[2]) - r
    var xmax = max(q0[0], q1[0]) + r
    var ymax = max(q0[1], q1[1]) + r
    var zmax = max(q0[2], q1[2]) + r
    return (xmin, ymin, zmin, xmax, ymax, zmax)

@always_inline
def curve_light_tube_area(curve: Curve_C) -> Float32:
    """Approximate lateral (side) surface area of a curve treated as a thin
    tube light: sum over each locally-linear piece of its lateral cylinder
    area (2*pi*avg_radius*piece_length). Used only to normalize NEE/light-
    sampler PDFs for emissive curves (AreaLight_C.kind==1) against the
    uniform-random-piece sampling used at runtime — see
    sample_area_light_uniform (sppm.mojo) and shading.mojo's curve NEE
    branch, which both pick pieces uniformly rather than area-weighted
    (mirroring the pre-existing mesh-light approximation of picking a
    uniform random triangle rather than area-weighting by triangle)."""
    var area = Float32(0.0)
    for piece in range(Int(curve.n_pieces)):
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var seg = q1 - q0
        var seg_len = sqrt(dot(seg, seg))
        area += Float32(2.0) * Float32(3.14159265358979323846) * ((r0 + r1) * Float32(0.5)) * seg_len
    return area

