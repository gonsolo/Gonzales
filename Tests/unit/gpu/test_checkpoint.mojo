from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.sys.info import size_of
from std.os import remove
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.gpu.checkpoint import Checkpoint, CheckpointConfig, checkpoint_adopt_seed
from gonzales.scene.description.pbrt_parser import ParsedScene_Mojo

comptime _Psc = Pointer[ParsedScene_Mojo, MutUntrackedOrigin]


def _scene() -> _Psc:
    var p = unsafe_alloc[ParsedScene_Mojo](1)
    var raw = p.unsafe_bitcast[UInt8]()
    for k in range(size_of[ParsedScene_Mojo]()):
        raw[unsafe_offset=k] = UInt8(0)
    var c2w = unsafe_alloc[Float32](16)
    for k in range(16):
        c2w[unsafe_offset=k] = Float32(k)
    p[unsafe_offset=0].camera_to_world = c2w
    p[unsafe_offset=0].film_w = Int32(8); p[unsafe_offset=0].film_h = Int32(4)
    p[unsafe_offset=0].samples_per_pixel = Int32(16); p[unsafe_offset=0].rng_seed = UInt64(1234)
    return p


def _cfg(path: String) raises -> CheckpointConfig:
    var cfg = CheckpointConfig()
    cfg.path = path
    cfg.scene = path + ".pbrt"
    with open(cfg.scene, "w") as f:
        f.write("WorldBegin\n")
    return cfg^


def _rm(path: String) raises:
    remove(path)
    remove(path + ".pbrt")


def _write(path: String, psc: _Psc) raises:
    var ck = Checkpoint(_cfg(path), psc, "test")
    ck.add("passes", "16")
    ck.begin_write(6)
    var a = unsafe_alloc[Float32](5)
    for k in range(5):
        a[unsafe_offset=k] = Float32(k) * Float32(0.5)
    ck.io("a", a, 5)
    var on = True
    ck.flag("on", on)
    var lam = Float32(0.25)
    ck.scalar("lambda", lam)
    ck.end_write(6)
    a.unsafe_free()


def test_round_trip() raises:
    var path = String("/tmp/gz_ckpt_test_rt.ckpt")
    var psc = _scene()
    _write(path, psc)
    var cfg = _cfg(path)
    cfg.resume = path
    var ck = Checkpoint(cfg, psc, "test")
    ck.add("passes", "16")
    assert_equal(ck.resume(), 6)
    var a = unsafe_alloc[Float32](5)
    ck.io("a", a, 5)
    var on = False
    ck.flag("on", on)
    var lam = Float32(0)
    ck.scalar("lambda", lam)
    ck.end_read()
    assert_true(on)
    assert_equal(lam, Float32(0.25))
    for k in range(5):
        assert_equal(a[unsafe_offset=k], Float32(k) * Float32(0.5))
    a.unsafe_free()
    _rm(path)


def test_header_mismatch_refuses() raises:
    var path = String("/tmp/gz_ckpt_test_mm.ckpt")
    var psc = _scene()
    _write(path, psc)
    psc[unsafe_offset=0].samples_per_pixel = Int32(32)
    var cfg = _cfg(path)
    cfg.resume = path
    var ck = Checkpoint(cfg, psc, "test")
    ck.add("passes", "16")
    var msg = String("")
    try:
        _ = ck.resume()
    except e:
        msg = String(e)
    _rm(path)
    assert_true("does not match" in msg and "spp" in msg)


def test_seed_adopted_without_cli_seed() raises:
    var path = String("/tmp/gz_ckpt_test_seed.ckpt")
    var psc = _scene()
    _write(path, psc)
    psc[unsafe_offset=0].rng_seed = UInt64(99)
    var cfg = _cfg(path)
    cfg.resume = path
    checkpoint_adopt_seed(cfg, psc)
    _rm(path)
    assert_equal(psc[unsafe_offset=0].rng_seed, UInt64(1234))


def test_truncated_refuses() raises:
    var path = String("/tmp/gz_ckpt_test_tr.ckpt")
    var psc = _scene()
    _write(path, psc)
    _ = external_call["truncate", Int32, Pointer[UInt8, MutUntrackedOrigin], Int](
        (path + "\0").unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), 200)
    var cfg = _cfg(path)
    cfg.resume = path
    var ck = Checkpoint(cfg, psc, "test")
    ck.add("passes", "16")
    var failed = False
    try:
        _ = ck.resume()
        var a = unsafe_alloc[Float32](5)
        ck.io("a", a, 5)
        ck.end_read()
    except:
        failed = True
    _rm(path)
    assert_true(failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
