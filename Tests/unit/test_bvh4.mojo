from std.math import abs, sqrt
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.geometry import Point3f, Vec3f
from gonzales.materials import MatKind
from gonzales.primitives import Ray, Intersection
from gonzales.bvh import (
    BVH4, build_bvh4, free_bvh4, traverse_bvh2_core, traverse_bvh4_core, any_hit_bvh2_core, any_hit_bvh4_core,
)
from _scene_fixture import make_triangle_scene

# BVH4 (a collapse of the BVH2) must give the same closest hit and the same
# any-hit answer as the BVH2 traversal on a real build_bvh2-built tree.

struct _Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32(Int((self.s >> 40) & UInt64(0xFFFFFF))) / Float32(16777216.0)


def _random_scene(n: Int, mut r: _Lcg, extent: Float32) -> List[Point3f]:
    var verts = List[Point3f]()
    for _ in range(n):
        var cx = r.next() * extent
        var cy = r.next() * extent
        var cz = r.next() * extent
        var sz = Float32(0.05) + r.next() * 0.6
        for _ in range(3):
            verts.append(Point3f(cx + (r.next() - 0.5) * sz, cy + (r.next() - 0.5) * sz, cz + (r.next() - 0.5) * sz))
    return verts^


def _compare(n_tris: Int, n_rays: Int, seed: UInt64, min_hits: Int = 0, extent: Float32 = 10.0, alpha: Float32 = 1.0) raises:
    var r = _Lcg(seed)
    var fx = make_triangle_scene(_random_scene(n_tris, r, extent))
    fx.meshes[unsafe_offset=0].alpha_const = alpha     # < 1: stochastic cut-out, decided by a hash of the ray
    var b4 = build_bvh4(fx.bvh_nodes, n_tris * 2 + 4, fx.prim_ids, n_tris, fx.meshes, fx.materials)
    assert_true(b4[1] > 0)
    var bvh4 = b4[0]
    var res2 = unsafe_alloc[Intersection](1)
    var res4 = unsafe_alloc[Intersection](1)
    var n_hit = 0
    var n_exact = 0
    var n_tie = 0
    var bad = 0
    for i in range(n_rays):
        var org = Point3f((r.next() * 1.4 - 0.2) * extent, (r.next() * 1.4 - 0.2) * extent, (r.next() * 1.4 - 0.2) * extent)
        var dx = r.next() * 2.0 - 1.0
        var dy = r.next() * 2.0 - 1.0
        var dz = r.next() * 2.0 - 1.0
        # Every 8th ray is axis-aligned (rdir = +-inf), the slab test's edge case.
        if i % 8 == 0:
            dx = 0.0
            dy = 0.0
            dz = 1.0
        elif i % 8 == 1:
            dx = -1.0
            dy = 0.0
            dz = 0.0
        var len = sqrt(dx * dx + dy * dy + dz * dz)
        if len < Float32(1e-3):
            continue
        var ray = Ray(org, Vec3f(dx / len, dy / len, dz / len))
        var tmax = Float32(1e30) if i % 3 != 0 else r.next() * 1.2 * extent
        traverse_bvh2_core(fx.bvh_nodes, fx.prim_ids, fx.meshes, fx.curves, ray, tmax, res2)
        traverse_bvh4_core(bvh4, fx.prim_ids, fx.meshes, fx.curves, ray, tmax, res4)
        var a = res2[unsafe_offset=0]
        var b = res4[unsafe_offset=0]
        if a.hit != b.hit:
            bad += 1
            continue
        if a.hit != Int8(0):
            n_hit += 1
            if abs(a.tHit - b.tHit) > Float32(1e-4) * a.tHit:
                bad += 1
                continue
            if a.primId.id2 == b.primId.id2:
                n_exact += 1
            elif a.tHit == b.tHit:
                n_tie += 1      # coincident triangles at the same distance: either may win
            else:
                bad += 1        # a different triangle at a different distance is a real mismatch
                continue
        var s2 = any_hit_bvh2_core(fx.bvh_nodes, fx.prim_ids, fx.meshes, fx.curves, ray, tmax)
        var s4 = any_hit_bvh4_core(bvh4, fx.prim_ids, fx.meshes, fx.curves, ray, tmax)
        if s2 != s4:
            bad += 1
    print("bvh4 vs bvh2:", n_tris, "tris,", n_rays, "rays,", n_hit, "hits,", n_exact, "identical,", n_tie, "ties,", b4[1], "bvh4 nodes,", bad, "mismatches")
    assert_equal(bad, 0)
    assert_true(n_hit >= min_hits)
    res2.unsafe_free()
    res4.unsafe_free()
    free_bvh4(bvh4)


def test_bvh4_matches_bvh2_large() raises:
    _compare(5000, 100000, UInt64(12345), 5000)


def test_bvh4_matches_bvh2_tiny_trees() raises:
    # Root-is-a-leaf and 2-3 level trees exercise the empty-lane paths.
    _compare(1, 40000, UInt64(1), 10, 1.0)
    _compare(2, 40000, UInt64(2), 10, 1.0)
    _compare(3, 40000, UInt64(3), 10, 1.0)
    _compare(7, 40000, UInt64(4), 50, 2.0)
    _compare(40, 40000, UInt64(5), 300, 4.0)


def test_bvh4_alpha_cutout_matches_bvh2() raises:
    # A half-transparent mesh takes the alpha_killed path on every hit.
    _compare(300, 60000, UInt64(77), 100, 3.0, 0.5)


def test_bvh4_shadow_rays_skip_interface_material() raises:
    # pbrt's "interface" material only marks a medium boundary: shadow rays that
    # are given the material table must pass through it, in both trees; without
    # the table the primitive blocks as before.
    var r = _Lcg(UInt64(5150))
    var fx = make_triangle_scene(_random_scene(200, r, 3.0))
    fx.materials[unsafe_offset=0].type = MatKind.interface
    var b4 = build_bvh4(fx.bvh_nodes, 404, fx.prim_ids, 200, fx.meshes, fx.materials)
    var bvh4 = b4[0]
    var n_blocked = 0
    for i in range(20000):
        var org = Point3f(r.next() * 3.0, r.next() * 3.0, r.next() * 3.0)
        var dx = r.next() * 2.0 - 1.0
        var dy = r.next() * 2.0 - 1.0
        var dz = r.next() * 2.0 - 1.0
        var len = sqrt(dx * dx + dy * dy + dz * dz)
        if len < Float32(1e-3):
            continue
        var ray = Ray(org, Vec3f(dx / len, dy / len, dz / len))
        var plain2 = any_hit_bvh2_core(fx.bvh_nodes, fx.prim_ids, fx.meshes, fx.curves, ray, Float32(100.0))
        var plain4 = any_hit_bvh4_core(bvh4, fx.prim_ids, fx.meshes, fx.curves, ray, Float32(100.0))
        assert_equal(plain2, plain4)
        if plain2:
            n_blocked += 1
        var skip2 = any_hit_bvh2_core(fx.bvh_nodes, fx.prim_ids, fx.meshes, fx.curves, ray, Float32(100.0), materials=fx.materials)
        var skip4 = any_hit_bvh4_core(bvh4, fx.prim_ids, fx.meshes, fx.curves, ray, Float32(100.0), materials=fx.materials)
        assert_true(not skip2)
        assert_true(not skip4)
    assert_true(n_blocked > 500)
    free_bvh4(bvh4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
