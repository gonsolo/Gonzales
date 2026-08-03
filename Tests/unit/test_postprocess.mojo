from std.math import abs
from std.collections import List
from std.testing import assert_true, TestSuite
from gonzales.postprocess import denoise

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# `denoise` is a multi-pass à-trous wavelet filter guided by albedo/normal/
# depth/luminance edge-stopping terms (see the doc comment above it in
# postprocess.mojo). It only needs small hand-built buffers, so it's fully
# testable without a rendered image or GPU context. `write_image` is not
# covered here: it's a thin external_call into the OpenImageIO C bridge
# with no logic on the Mojo side.

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

# ── Matches the à-trous B3-spline closed form ────────────────────────────────
# denoise() switched from a single-pass bilateral filter (spatial Gaussian x
# range weights) to a multi-pass à-trous wavelet filter, unified with the GPU
# denoiser 2026-08-03 (see project_gpu_denoiser_energy_bug.md memory) --
# among other things this repurposed parameter 8 from "sigma_s, a spatial
# Gaussian std-dev" to "sigma_l, a luminance/variance tolerance", so the old
# test_denoise_matches_spatial_gaussian_average above no longer describes
# what the code does (correctly failed after the rewrite, replaced here
# rather than deleted).

def test_denoise_matches_atrous_b3_spline_kernel() raises:
    """With albedo/normal/depth identical everywhere AND sigma_l set huge
    enough that the luminance/variance term is ~1 regardless of value
    spread, a single pass (n_passes=1, step=1) collapses to the FIXED
    separable B3-spline kernel {0.375, 0.25, 0.0625} at tap distance
    {0, 1, 2} -- computed here independently from denoise's internals,
    not by re-reading them. For a single-row image only the dx-direction
    weights matter: dy=0 is the only in-bounds row, and its hy factor is
    common to every tap, so it cancels out of the normalized average."""
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

    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, output.unsafe_ptr(), Int32(1),
            Float32(1.0e6), Float32(0.2), Float32(0.3), Float32(0.05))

    var weighted_sum = Float32(0)
    var weight_total = Float32(0)
    for dx in range(-2, 3):
        var adx = dx if dx >= 0 else -dx
        var wt = Float32(0.375) if adx == 0 else (Float32(0.25) if adx == 1 else Float32(0.0625))
        weighted_sum += wt * vals[2 + dx]
        weight_total += wt
    var expected = weighted_sum / weight_total

    assert_true(_close(output[2 * 3 + 0], expected))

# ── Energy conservation: regression guard for the real bug this replaced ────

def test_denoise_conserves_energy_for_isolated_bright_pixel() raises:
    """Direct regression test for a real, measured bug (see
    project_gpu_denoiser_energy_bug.md): keying the luminance edge-stopping
    weight to only the CENTRE pixel's variance makes w(p,q) != w(q,p), and
    since the filter normalizes by its own weight sum, that asymmetry
    DESTROYS energy instead of moving it -- a bright pixel accepts dim
    neighbours and dims, while those neighbours (low variance) reject it
    and never brighten. Fixed via min(var_p,var_q) in _atrous_tap_weight.
    One bright pixel among dim ones, uniform albedo/normal/depth so
    spatial mixing is unobstructed: total output energy must stay close
    to total input energy, not collapse toward the dim majority.

    The spike is 3x its neighbours, not more: denoise() ALSO runs a
    firefly pre-clamp (_clamp_fireflies) ahead of à-trous, which by design
    suppresses a pixel more than _FIREFLY_CLAMP_K(=4)x its neighbours'
    max -- a genuinely separate mitigation for single-pixel MC noise
    spikes, not the bug under test here. Staying under that threshold
    isolates the à-trous weight-symmetry property from the pre-clamp."""
    var w = Int32(7)
    var h = Int32(1)
    var vals = [Float32(5.0), Float32(5.0), Float32(5.0), Float32(15.0), Float32(5.0), Float32(5.0), Float32(5.0)]
    var beauty: List[Float32] = []
    var albedo: List[Float32] = []
    var normals: List[Float32] = []
    var depth: List[Float32] = []
    for i in range(7):
        beauty.append(vals[i]); beauty.append(vals[i]); beauty.append(vals[i])
        albedo.append(Float32(0.0)); albedo.append(Float32(0.0)); albedo.append(Float32(0.0))
        normals.append(Float32(0.0)); normals.append(Float32(0.0)); normals.append(Float32(1.0))
        depth.append(Float32(1.0))
    var output = _zeros(7 * 3)

    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, output.unsafe_ptr(), Int32(2),
            Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

    var in_sum = Float32(0)
    var out_sum = Float32(0)
    for i in range(7):
        in_sum += vals[i]
        out_sum += output[i * 3 + 0]
    # A generous band, not a tight equality: some redistribution toward the
    # dim majority is the FILTER'S JOB (that's denoising). The bug lost
    # ~32% of total energy outright; conservation within 15% here would
    # have failed hard under the old asymmetric weight.
    assert_true(out_sum > in_sum * Float32(0.85))
    assert_true(out_sum < in_sum * Float32(1.15))

# ── Multi-pass dilation actually reaches farther than a single pass ─────────

def test_denoise_more_passes_reduces_variance_more() raises:
    """à-trous's whole point is that pass i dilates its taps by step=2^i, so
    a handful of cheap 5x5 passes reach a large effective radius -- more
    passes should smooth a mildly-varying signal harder.

    Deliberately NOT a large isolated spike far from a flat region: that
    was this test's first draft, and it failed for an instructive reason,
    not a bug. The luminance edge-stopping weight is min(var_p,var_q)-keyed
    (see _atrous_tap_weight); a pixel whose OWN local neighbourhood is
    perfectly uniform has var_p EXACTLY 0, which collapses sigma_l2 to the
    1e-6 floor regardless of sigma_l, correctly rejecting a distant,
    genuinely-different value no matter how many passes run -- that is the
    filter WORKING (an edge-aware denoiser is supposed to refuse to blend
    unrelated brightness levels; unbounded diffusion would defeat the
    entire technique, which is why the separate, deliberately-biased
    firefly clamp exists for real noise spikes instead).

    A mildly-varying row sidesteps that: every pixel's local window has
    genuine nonzero variance, so the luminance term stays engaged
    (not floor-limited) and mixing is legitimate, letting variance
    actually shrink with more passes -- exactly the property useful for
    catching an accidentally-inert loop (e.g. an off-by-one collapsing
    every pass to a no-op, or ping/pong writing back to its own input)."""
    var w = Int32(9)
    var h = Int32(1)
    var vals = [Float32(1.0), Float32(1.2), Float32(0.9), Float32(1.1), Float32(1.0),
                Float32(0.95), Float32(1.05), Float32(0.85), Float32(1.15)]
    var beauty: List[Float32] = []
    var albedo: List[Float32] = _zeros(9 * 3)
    var normals: List[Float32] = []
    var depth: List[Float32] = []
    for i in range(9):
        beauty.append(vals[i]); beauty.append(vals[i]); beauty.append(vals[i])
        normals.append(Float32(0.0)); normals.append(Float32(0.0)); normals.append(Float32(1.0))
        depth.append(Float32(1.0))

    var out1 = _zeros(9 * 3)
    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, out1.unsafe_ptr(), Int32(1),
            Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))
    var out5 = _zeros(9 * 3)
    denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(), normals.unsafe_ptr(), depth.unsafe_ptr(),
            w, h, out5.unsafe_ptr(), Int32(5),
            Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

    var mean1 = Float32(0)
    var mean5 = Float32(0)
    for i in range(9):
        mean1 += out1[i * 3 + 0]
        mean5 += out5[i * 3 + 0]
    mean1 /= Float32(9); mean5 /= Float32(9)
    var var1 = Float32(0)
    var var5 = Float32(0)
    for i in range(9):
        var d1 = out1[i * 3 + 0] - mean1
        var d5 = out5[i * 3 + 0] - mean5
        var1 += d1 * d1
        var5 += d5 * d5

    assert_true(var5 < var1)

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
