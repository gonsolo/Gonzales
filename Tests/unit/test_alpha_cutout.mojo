# Unit tests for alpha_killed (geometry.mojo), the one `Shape "texture
# alpha"` / `"float alpha"` test every BVH triangle-hit site calls.

from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import Vec3f
from gonzales.primitives import TriangleMesh, alpha_killed


def _mesh(mask: Pointer[UInt8, MutUntrackedOrigin], w: Int32, h: Int32,
          alpha_const: Float32) -> TriangleMesh:
    """One triangle with no UVs, so pbrt's default (0,0) (1,0) (1,1)
    parameterisation applies: barycentrics (bu, bv) land at
    uv = (bu + bv, bv)."""
    var vidx = unsafe_alloc[Int64](3)
    vidx[unsafe_offset=0] = 0; vidx[unsafe_offset=1] = 1; vidx[unsafe_offset=2] = 2
    return TriangleMesh(
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(), vidx,
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        mask, w, h, alpha_const,
    )


def _no_mask() -> Pointer[UInt8, MutUntrackedOrigin]:
    return Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()


def test_constant_alpha_keeps_opaque_and_drops_transparent() raises:
    var o = Vec3f(0.1, 0.2, 0.3)
    var d = Vec3f(0.0, 0.0, 1.0)
    assert_false(alpha_killed(_mesh(_no_mask(), 0, 0, 1.0), 0, 1, 2, 0.2, 0.2, o, d, 7))
    assert_true(alpha_killed(_mesh(_no_mask(), 0, 0, 0.0), 0, 1, 2, 0.2, 0.2, o, d, 7))


def test_constant_fractional_alpha_keeps_that_fraction_of_rays() raises:
    """pbrt's stochastic rule: a hit at alpha a survives with probability a,
    decided by a hash of the ray -- so over many distinct rays the kept
    fraction converges to a."""
    var mesh = _mesh(_no_mask(), 0, 0, 0.3)
    var kept = 0
    comptime N = 20000
    for i in range(N):
        var o = Vec3f(Float32(i) * 0.001, Float32(i % 97) * 0.01, 0.0)
        if not alpha_killed(mesh, 0, 1, 2, 0.2, 0.2, o, Vec3f(0.0, 0.0, 1.0), 3):
            kept += 1
    var frac = Float32(kept) / Float32(N)
    assert_true(frac > 0.28 and frac < 0.32)


def test_mask_cuts_out_by_texel_with_v_flip() raises:
    """A 2x2 mask whose TOP row (image row 0) is opaque and bottom row
    transparent. pbrt flips v, so the opaque row is the one at v near 1."""
    var mask = unsafe_alloc[UInt8](4)
    mask[unsafe_offset=0] = 255; mask[unsafe_offset=1] = 255   # image row 0 (top)
    mask[unsafe_offset=2] = 0;   mask[unsafe_offset=3] = 0     # image row 1 (bottom)
    var mesh = _mesh(mask, 2, 2, 1.0)
    var o = Vec3f(0.5, 0.5, -1.0)
    var d = Vec3f(0.0, 0.0, 1.0)
    # Sample on texel-row centres (flipped v = 0.25 / 0.75) so the bilinear
    # filter reads one row only and the answer is deterministic.
    # bv = 0.75 -> v = 0.75, flipped to 0.25: the opaque top row.
    assert_false(alpha_killed(mesh, 0, 1, 2, 0.1, 0.75, o, d, 1))
    # bv = 0.25 -> v = 0.25, flipped to 0.75: the transparent bottom row.
    assert_true(alpha_killed(mesh, 0, 1, 2, 0.3, 0.25, o, d, 1))
    mask.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
