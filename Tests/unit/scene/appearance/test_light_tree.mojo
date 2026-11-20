from std.math import abs, sqrt
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.core.geometry import Vec3f, cross
from gonzales.core.rng import PCG32
from gonzales.scene.appearance.light_tree import (LightBounds, build_light_tree, light_tree_sample, light_tree_pmf,
                                                   light_tree_importance, light_tree_emitter, light_tree_light)


def _triangle(a: Vec3f, b: Vec3f, c: Vec3f, phi: Float32, two_sided: Bool) -> LightBounds:
    var n = cross(b - a, c - a).normalize()
    return LightBounds(Vec3f(min(a.x, min(b.x, c.x)), min(a.y, min(b.y, c.y)), min(a.z, min(b.z, c.z))),
                       Vec3f(max(a.x, max(b.x, c.x)), max(a.y, max(b.y, c.y)), max(a.z, max(b.z, c.z))),
                       n, phi, 1.0, 0.0, two_sided)


def _flat(n: Int) -> List[Int]:
    var b = List[Int]()
    for i in range(n + 1):
        b.append(i)
    return b^


def _blob(words: List[UInt32]) -> Pointer[UInt32, MutUntrackedOrigin]:
    var t = unsafe_alloc[UInt32](len(words))
    for i in range(len(words)):
        t[unsafe_offset=i] = words[i]
    return t


def _scene(n: Int, mut pcg: PCG32) -> List[LightBounds]:
    var b = List[LightBounds]()
    for i in range(n):
        var o = Vec3f(pcg.next_float() * 10 - 5, pcg.next_float() * 10 - 5, pcg.next_float() * 10 - 5)
        var a = o + Vec3f(pcg.next_float(), pcg.next_float(), pcg.next_float()) * 0.5
        var c = o + Vec3f(pcg.next_float(), pcg.next_float(), pcg.next_float()) * 0.5
        b.append(_triangle(o, a, c, Float32(0) if i % 17 == 3 else pcg.next_float() * 4 + 0.1, i % 5 == 0))
    return b^


def test_pmf_sums_to_one_and_matches_sample() raises:
    var pcg = PCG32(7, 1)
    var bounds = _scene(300, pcg)
    var t = _blob(build_light_tree(bounds, _flat(len(bounds))))
    var single = List[Pointer[UInt32, MutUntrackedOrigin]]()
    for e in range(len(bounds)):
        var one = List[LightBounds]()
        one.append(bounds[e])
        single.append(_blob(build_light_tree(one, _flat(1))))
    for _ in range(40):
        var p = Vec3f(pcg.next_float() * 14 - 7, pcg.next_float() * 14 - 7, pcg.next_float() * 14 - 7)
        var n = Vec3f(pcg.next_float() - 0.5, pcg.next_float() - 0.5, pcg.next_float() - 0.5).normalize()
        var total = Float32(0)
        for e in range(len(bounds)):
            var pm = light_tree_pmf(t, e, p, n)
            if bounds[e].phi == 0:
                assert_equal(pm, Float32(0))
            elif light_tree_importance(single[e], 0, p, n) > 0:
                assert_true(pm > 0, "emitter " + String(e) + " can light p but has pmf 0")
            total += pm
        # Below 1 where a node's children both turn out to be unable to light p: pbrt's sampler fails there too.
        assert_true(total <= 1 + 1e-4 and total > 0.5, "pmf sum " + String(total))
        for _ in range(20):
            var (e, pm) = light_tree_sample(t, p, n, pcg.next_float())
            if e >= 0:
                var back = light_tree_pmf(t, e, p, n)
                assert_true(abs(pm - back) <= 1e-5 * max(pm, Float32(1e-3)), "sample pmf " + String(pm) + " vs " + String(back))
    t.unsafe_free()
    for e in range(len(single)):
        single[e].unsafe_free()


def test_frequencies_follow_pmf() raises:
    var pcg = PCG32(11, 3)
    var bounds = _scene(24, pcg)
    var t = _blob(build_light_tree(bounds, _flat(len(bounds))))
    var p = Vec3f(0.3, -0.2, 6.5)
    var n = Vec3f(0.0, 0.0, 0.0)
    var counts = List[Int](length=len(bounds), fill=0)
    var trials = 50000
    for i in range(trials):
        var (e, _) = light_tree_sample(t, p, n, (Float32(i) + 0.5) / Float32(trials))
        if e >= 0:
            counts[e] += 1
    for e in range(len(bounds)):
        var expect = light_tree_pmf(t, e, p, n)
        var got = Float32(counts[e]) / Float32(trials)
        assert_true(abs(got - expect) < 4e-3, "emitter " + String(e) + ": " + String(got) + " vs " + String(expect))
    t.unsafe_free()


def test_single_sided_light_is_seen_only_from_its_front() raises:
    var bounds = List[LightBounds]()
    bounds.append(_triangle(Vec3f(0.0, 0.0, 0.0), Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0), 1.0, False))
    bounds.append(_triangle(Vec3f(5.0, 0.0, 0.0), Vec3f(6.0, 0.0, 0.0), Vec3f(5.0, 1.0, 0.0), 1.0, False))
    var t = _blob(build_light_tree(bounds, _flat(len(bounds))))
    var zero = Vec3f(0.0, 0.0, 0.0)
    assert_true(light_tree_pmf(t, 0, Vec3f(0.3, 0.3, 2.0), zero) > 0.5)
    assert_equal(light_tree_pmf(t, 0, Vec3f(0.3, 0.3, -2.0), zero), Float32(0))
    assert_equal(light_tree_pmf(t, 1, Vec3f(5.3, 0.3, -2.0), zero), Float32(0))
    # Grazing, just in front of the plane: quantization must not cull it.
    assert_true(light_tree_pmf(t, 0, Vec3f(3.0, 0.3, 0.01), zero) > 0)
    t.unsafe_free()


def test_emitters_map_to_lights_and_triangles() raises:
    var pcg = PCG32(5, 9)
    var bounds = _scene(10, pcg)
    var base = List[Int]()
    base.append(0); base.append(4); base.append(4); base.append(9); base.append(10)
    var t = _blob(build_light_tree(bounds, base))
    for l in range(4):
        for tri in range(base[l + 1] - base[l]):
            var e = light_tree_emitter(t, l, tri)
            assert_equal(e, base[l] + tri)
            var (l2, tri2) = light_tree_light(t, e)
            assert_equal(l2, l)
            assert_equal(tri2, tri)
    t.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
