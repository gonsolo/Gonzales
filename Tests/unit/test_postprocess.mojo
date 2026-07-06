from std.math import abs, exp
from std.collections import List
from std.testing import assert_true, TestSuite
from gonzales.postprocess import denoise

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# `denoise` is a joint bilateral filter guided by albedo/normal/depth edge
# stopping terms (see the doc comment above it in postprocess.mojo). It only
# needs small hand-built buffers, so it's fully testable without a rendered
# image or GPU context. `write_image` is not covered here: it's a thin
# external_call into the OpenImageIO C bridge with no logic on the Mojo side.

def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(Float32(0))
    return out^

# ── Uniform image is a fixed point ──────────────────────────────────────────

def test_denoise_uniform_image_is_unchanged() raises:
    """A weighted average of identical values must equal that value, no
    matter what the spatial/albedo/normal/depth weights work out to."""
    var w = Int32(4)
    var h = Int32(3)
    var n = 12
    var beauty: List[Float32] = []
    var albedo: List[Float32] = []
    var normals: List[Float32] = []
    var depth: List[Float32] = []
    for _ in range(n):
        beauty.append(Float32(5.0)); beauty.append(Float32(3.0)); beauty.append(Float32(1.0))
        albedo.append(Float32(0.2)); albedo.append(Float32(0.4)); albedo.append(Float32(0.6))
        normals.append(Float32(0.0)); normals.append(Float32(0.0)); normals.append(Float32(1.0))
        depth.append(Float32(2.0))
    var output = _zeros(n * 3)

    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, output.unsafe_ptr(), Int32(2),
            Float32(3.0), Float32(0.2), Float32(0.3), Float32(0.05))

    for i in range(n):
        assert_true(_close(output[i * 3 + 0], Float32(5.0)))
        assert_true(_close(output[i * 3 + 1], Float32(3.0)))
        assert_true(_close(output[i * 3 + 2], Float32(1.0)))

# ── Matches the documented spatial Gaussian closed form ─────────────────────

def test_denoise_matches_spatial_gaussian_average() raises:
    """With albedo/normal/depth identical everywhere, the edge-stopping
    terms all collapse to 1 and the result is a pure spatial Gaussian
    average — computed here independently from the module's own doc
    comment (`weight = exp(-dist^2 / (2*sigma_s^2))`), not by re-reading
    denoise's internals."""
    var w = Int32(5)
    var h = Int32(1)
    var vals = [Float32(1.0), Float32(2.0), Float32(10.0), Float32(4.0), Float32(5.0)]
    var beauty: List[Float32] = []
    var albedo: List[Float32] = []
    var normals: List[Float32] = []
    var depth: List[Float32] = []
    for i in range(5):
        beauty.append(vals[i]); beauty.append(vals[i]); beauty.append(vals[i])
        albedo.append(Float32(0.0)); albedo.append(Float32(0.0)); albedo.append(Float32(0.0))
        normals.append(Float32(0.0)); normals.append(Float32(0.0)); normals.append(Float32(1.0))
        depth.append(Float32(1.0))
    var output = _zeros(5 * 3)

    var sigma_s = Float32(1.5)
    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, output.unsafe_ptr(), Int32(2),
            sigma_s, Float32(0.2), Float32(0.3), Float32(0.05))

    var inv2ss = Float32(1) / (Float32(2) * sigma_s * sigma_s)
    var weighted_sum = Float32(0)
    var weight_total = Float32(0)
    for dx in range(-2, 3):
        var wt = exp(-Float32(dx * dx) * inv2ss)
        weighted_sum += wt * vals[2 + dx]
        weight_total += wt
    var expected = weighted_sum / weight_total

    assert_true(_close(output[2 * 3 + 0], expected))

# ── Albedo edge stopping preserves flat regions across a hard boundary ──────

def test_denoise_albedo_edge_preserves_flat_regions() raises:
    """Two flat regions with very different albedo, joined at a boundary.
    A tiny sigma_r makes the cross-boundary weight underflow to ~0, so each
    side's output must stay at its own flat value instead of blending
    toward the average of both sides."""
    var w = Int32(6)
    var h = Int32(1)
    var beauty_vals = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(10.0), Float32(10.0), Float32(10.0)]
    var albedo_vals = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0), Float32(1.0), Float32(1.0)]
    var beauty: List[Float32] = []
    var albedo: List[Float32] = []
    var normals: List[Float32] = []
    var depth: List[Float32] = []
    for i in range(6):
        beauty.append(beauty_vals[i]); beauty.append(beauty_vals[i]); beauty.append(beauty_vals[i])
        albedo.append(albedo_vals[i]); albedo.append(albedo_vals[i]); albedo.append(albedo_vals[i])
        normals.append(Float32(0.0)); normals.append(Float32(0.0)); normals.append(Float32(1.0))
        depth.append(Float32(1.0))
    var output = _zeros(6 * 3)

    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, output.unsafe_ptr(), Int32(2),
            Float32(5.0), Float32(0.02), Float32(1.0), Float32(1.0))

    # Pixel 2 sits right at the boundary; without albedo rejection its large
    # sigma_s=5 spatial window would strongly mix in the value-10 side.
    assert_true(_close(output[2 * 3 + 0], Float32(0.0)))
    assert_true(_close(output[3 * 3 + 0], Float32(10.0)))

# ── Normal discontinuity downweights a facing-away neighbor ────────────────

def test_denoise_normal_discontinuity_downweights_neighbor() raises:
    """Cos-based normal term: 1-dot(n0,n1) is 2 for an opposite-facing
    neighbor. With a small sigma_n that neighbor's weight underflows to
    ~0, so a bright outlier just past a normal discontinuity must not
    pull the result up."""
    var w = Int32(3)
    var h = Int32(1)
    var beauty: List[Float32] = [Float32(5.0), Float32(5.0), Float32(5.0),
                                  Float32(5.0), Float32(5.0), Float32(5.0),
                                  Float32(50.0), Float32(50.0), Float32(50.0)]
    var albedo: List[Float32] = _zeros(9)
    var normals: List[Float32] = [Float32(0.0), Float32(0.0), Float32(1.0),
                                   Float32(0.0), Float32(0.0), Float32(1.0),
                                   Float32(0.0), Float32(0.0), Float32(-1.0)]
    var depth: List[Float32] = [Float32(1.0), Float32(1.0), Float32(1.0)]
    var output = _zeros(9)

    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, output.unsafe_ptr(), Int32(1),
            Float32(2.0), Float32(0.5), Float32(0.05), Float32(1.0))

    # Pixel 1's two contributing neighbors (itself and pixel 0) are both
    # exactly 5.0, so if pixel 2 (the opposite-normal outlier) is properly
    # rejected the result is exactly 5.0 regardless of the spatial weights.
    assert_true(_close(output[1 * 3 + 0], Float32(5.0)))

# ── Depth discontinuity downweights a distant neighbor ──────────────────────

def test_denoise_depth_discontinuity_downweights_neighbor() raises:
    """Depth term is relative: ((d1-d0)/d0)^2. A neighbor 100x farther away
    than the center pixel must be rejected under a small sigma_d, the same
    way the normal test rejects an opposite-facing neighbor above."""
    var w = Int32(3)
    var h = Int32(1)
    var beauty: List[Float32] = [Float32(5.0), Float32(5.0), Float32(5.0),
                                  Float32(5.0), Float32(5.0), Float32(5.0),
                                  Float32(50.0), Float32(50.0), Float32(50.0)]
    var albedo: List[Float32] = _zeros(9)
    var normals: List[Float32] = [Float32(0.0), Float32(0.0), Float32(1.0),
                                   Float32(0.0), Float32(0.0), Float32(1.0),
                                   Float32(0.0), Float32(0.0), Float32(1.0)]
    var depth: List[Float32] = [Float32(1.0), Float32(1.0), Float32(100.0)]
    var output = _zeros(9)

    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, output.unsafe_ptr(), Int32(1),
            Float32(2.0), Float32(0.5), Float32(1.0), Float32(0.1))

    assert_true(_close(output[1 * 3 + 0], Float32(5.0)))

# ── Boundary handling: out-of-window neighbors are skipped, not crashed ─────

def test_denoise_single_pixel_returns_input_unchanged() raises:
    """A 1x1 image with a radius larger than the image: every neighbor
    offset except (0,0) is out of bounds and must be skipped (not read
    out-of-bounds / not crash), leaving the single pixel as its own
    unweighted result."""
    var beauty: List[Float32] = [Float32(7.0), Float32(8.0), Float32(9.0)]
    var albedo: List[Float32] = [Float32(0.1), Float32(0.2), Float32(0.3)]
    var normals: List[Float32] = [Float32(0.0), Float32(0.0), Float32(1.0)]
    var depth: List[Float32] = [Float32(2.0)]
    var output = _zeros(3)

    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            Int32(1), Int32(1), output.unsafe_ptr(), Int32(3),
            Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0))

    assert_true(_close(output[0], Float32(7.0)))
    assert_true(_close(output[1], Float32(8.0)))
    assert_true(_close(output[2], Float32(9.0)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
