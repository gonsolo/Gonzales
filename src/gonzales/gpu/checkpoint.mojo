"""Checkpoint and resume for the GPU drivers (PT, VCM, SPPM).

A checkpoint is the integrator's accumulated state after `done` passes (samples for PT). It is written to <file>.tmp,
synced and renamed over <file>, so a crash while writing keeps the last good one. Layout:
    "GZCKPT01" | Int header length | header text ("key=value" lines) | Int done | sections | "GZCKEND1"
    section: Int name length | name | Int byte count | bytes
A resume compares the header line by line and refuses on any difference. A driver reads its sections with the same
`io` calls that wrote them (`Checkpoint.writing` picks the direction), so the order cannot drift.
"""
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.sys.info import size_of
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from gonzales.scene.description.pbrt_parser import ParsedScene_Mojo

comptime _P = Pointer[UInt8, MutUntrackedOrigin]
comptime _MAGIC = "GZCKPT01"
comptime _END = "GZCKEND1"


def _cstr(s: String) -> _P:
    var n = s.byte_length()
    var c = unsafe_alloc[UInt8](n + 1)
    for k in range(n):
        c[unsafe_offset=k] = s.unsafe_ptr()[unsafe_offset=k]
    c[unsafe_offset=n] = UInt8(0)
    return c


def _fopen(path: String, mode: String) -> Int:
    var p = _cstr(path)
    var m = _cstr(mode)
    var f = external_call["fopen", Int, _P, _P](p, m)
    p.unsafe_free(); m.unsafe_free()
    return f


def _realpath(path: String) -> String:
    var p = _cstr(path)
    var r = external_call["realpath", Int, _P, Int](p, 0)
    p.unsafe_free()
    if r == 0:
        return path
    var out = String(unsafe_from_utf8_ptr=_P(unsafe_from_address=r).as_imm())
    external_call["free", NoneType, Int](r)
    return out


def _fnv(h: UInt64, p: _P, n: Int) -> UInt64:
    var x = h
    for k in range(n):
        x = (x ^ UInt64(p[unsafe_offset=k])) * UInt64(0x100000001B3)
    return x


struct CheckpointConfig(Copyable, Movable):
    """The command line's checkpoint flags. `every_s` < 0: only at the end; `stop_after` > 0: stop after that many
    passes (or samples) with a checkpoint and no image."""
    var path: String
    var resume: String
    var every_s: Float64
    var stop_after: Int
    var scene: String
    var seed_given: Bool

    def __init__(out self):
        self.path = String(""); self.resume = String(""); self.every_s = 600.0; self.stop_after = 0
        self.scene = String(""); self.seed_given = False

    def active(self) -> Bool:
        return self.path.byte_length() > 0 or self.resume.byte_length() > 0


def checkpoint_adopt_seed(cfg: CheckpointConfig, psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin]) raises:
    """Without --seed a render's seed comes from the clock: a resume takes the checkpoint's."""
    if cfg.resume.byte_length() == 0 or cfg.seed_given:
        return
    var c = Checkpoint(cfg, psc, "")
    var lines = c._read_header()
    c._close()
    for ref l in lines:
        if l.startswith("seed="):
            psc[unsafe_offset=0].rng_seed = UInt64(atol(String(l).removeprefix("seed=")))


struct Checkpoint(Movable):
    var cfg: CheckpointConfig
    var keys: List[String]
    var vals: List[String]
    var writing: Bool
    var f: Int
    var last_ns: Int
    var bytes: Int

    def __init__(out self, cfg: CheckpointConfig, psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin], integrator: String):
        self.cfg = cfg.copy()
        self.keys = List[String](); self.vals = List[String]()
        self.writing = False; self.f = 0; self.last_ns = perf_counter_ns(); self.bytes = 0
        if not cfg.active():
            return
        ref s = psc[unsafe_offset=0]
        # The scene file's bytes and what the parser built from them (includes and meshes show up in the counts).
        var h = UInt64(0xCBF29CE484222325)
        var size = 0
        try:
            var text = open(cfg.scene, "r").read()
            size = text.byte_length()
            h = _fnv(h, text.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), size)
        except:
            pass
        var counts = List[Int]()
        counts.append(Int(s.prim_count)); counts.append(Int(s.mesh_count)); counts.append(Int(s.sphere_count))
        counts.append(Int(s.curve_count)); counts.append(Int(s.instance_count)); counts.append(Int(s.area_light_count))
        counts.append(Int(s.medium_count))
        for c in counts:
            h = (h ^ UInt64(c)) * UInt64(0x100000001B3)
        h = _fnv(h, s.camera_to_world.unsafe_bitcast[UInt8](), 16 * 4)
        self.add("integrator", integrator)
        self.add("scene", _realpath(cfg.scene))
        self.add("scene_bytes", String(size))
        self.add("scene_hash", String(h))
        self.add("film", String(s.film_w) + "x" + String(s.film_h))
        self.add("spp", String(s.samples_per_pixel))
        self.add("maxdepth", String(s.max_depth))
        self.add("seed", String(s.rng_seed))
        self.add("clamp", String(s.sample_clamp()))

    def add(mut self, key: String, value: String):
        self.keys.append(key); self.vals.append(value)

    def enabled(self) -> Bool:
        return self.cfg.path.byte_length() > 0

    def due(self) -> Bool:
        return self.enabled() and self.cfg.every_s >= 0.0 and Float64(perf_counter_ns() - self.last_ns) >= self.cfg.every_s * 1.0e9

    def stop_here(self, done: Int) -> Bool:
        return self.cfg.stop_after > 0 and done >= self.cfg.stop_after

    def _header(self) -> String:
        var t = String("")
        for i in range(len(self.keys)):
            t += self.keys[i] + "=" + self.vals[i] + "\n"
        return t

    def _put(mut self, p: _P, n: Int) raises:
        if n > 0 and external_call["fwrite", Int, _P, Int, Int, Int](p, 1, n, self.f) != n:
            raise Error("checkpoint: write to " + self.cfg.path + ".tmp failed (disk full?)")
        self.bytes += n

    def _get(mut self, p: _P, n: Int) raises:
        if n > 0 and external_call["fread", Int, _P, Int, Int, Int](p, 1, n, self.f) != n:
            raise Error("checkpoint " + self.cfg.resume + " is truncated")

    def _int(mut self, v: Int = 0) raises -> Int:
        var p = unsafe_alloc[Int](1)
        p[unsafe_offset=0] = v
        if self.writing:
            self._put(p.unsafe_bitcast[UInt8](), size_of[Int]())
        else:
            self._get(p.unsafe_bitcast[UInt8](), size_of[Int]())
        var r = p[unsafe_offset=0]
        p.unsafe_free()
        return r

    def _str(mut self, s: String = String("")) raises -> String:
        var n = self._int(s.byte_length())
        if self.writing:
            self._put(s.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), n)
            return s
        if n < 0 or n > (1 << 24):
            raise Error("checkpoint " + self.cfg.resume + " is not a checkpoint")
        var c = unsafe_alloc[UInt8](n + 1)
        self._get(c, n)
        c[unsafe_offset=n] = UInt8(0)
        var out = String(unsafe_from_utf8_ptr=c.as_imm())
        c.unsafe_free()
        return out

    def _magic(mut self, m: String) raises:
        var c = _cstr(m)
        if self.writing:
            self._put(c, 8)
        else:
            var got = unsafe_alloc[UInt8](8)
            self._get(got, 8)
            var ok = True
            for k in range(8):
                ok = ok and got[unsafe_offset=k] == c[unsafe_offset=k]
            got.unsafe_free()
            if not ok:
                c.unsafe_free()
                raise Error("checkpoint " + self.cfg.resume + " is not a checkpoint or is incomplete")
        c.unsafe_free()

    def _close(mut self):
        if self.f != 0:
            _ = external_call["fclose", Int32, Int](self.f)
            self.f = 0

    def _read_header(mut self) raises -> List[String]:
        self.writing = False
        self.f = _fopen(self.cfg.resume, "rb")
        if self.f == 0:
            raise Error("checkpoint " + self.cfg.resume + " cannot be opened")
        self._magic(_MAGIC)
        var lines = List[String]()
        for l in self._str().split("\n"):
            if l.byte_length() > 0:
                lines.append(String(l))
        return lines^

    def resume(mut self) raises -> Int:
        """Opens --resume and checks its header against this render's; the done count, or 0 without --resume. The
        caller then reads its sections with `io` and calls `end_read`."""
        if self.cfg.resume.byte_length() == 0:
            return 0
        var lines = self._read_header()
        var why = String("")
        for i in range(len(self.keys)):
            var want = self.keys[i] + "=" + self.vals[i]
            var have = String("(missing)")
            for ref l in lines:
                if l.startswith(self.keys[i] + "="):
                    have = l
            if have != want:
                why += "\n  " + self.keys[i] + ": checkpoint has '" + String(have.removeprefix(self.keys[i] + "=")) + "', this render has '" + self.vals[i] + "'"
        if len(lines) != len(self.keys) and why.byte_length() == 0:
            why = "\n  the checkpoint has " + String(len(lines)) + " header entries, this render " + String(len(self.keys))
        if why.byte_length() > 0:
            self._close()
            raise Error("checkpoint " + self.cfg.resume + " does not match this render, refusing to resume:" + why)
        var done = self._int()
        print("Resuming from " + self.cfg.resume + " at " + String(done))
        return done

    def end_read(mut self) raises:
        self._magic(_END)
        self._close()
        self.last_ns = perf_counter_ns()

    def begin_write(mut self, done: Int) raises:
        self.writing = True
        self.bytes = 0
        self.f = _fopen(self.cfg.path + ".tmp", "wb")
        if self.f == 0:
            raise Error("checkpoint: cannot create " + self.cfg.path + ".tmp")
        self._magic(_MAGIC)
        _ = self._str(self._header())
        _ = self._int(done)

    def end_write(mut self, done: Int) raises:
        """Syncs <file>.tmp and renames it over <file>."""
        self._magic(_END)
        var ok = external_call["fflush", Int32, Int](self.f) == 0
        ok = ok and external_call["fsync", Int32, Int32](external_call["fileno", Int32, Int](self.f)) == 0
        ok = (external_call["fclose", Int32, Int](self.f) == 0) and ok
        self.f = 0
        if not ok:
            raise Error("checkpoint: writing " + self.cfg.path + ".tmp failed")
        var tmp = _cstr(self.cfg.path + ".tmp")
        var dst = _cstr(self.cfg.path)
        var rc = external_call["rename", Int32, _P, _P](tmp, dst)
        tmp.unsafe_free(); dst.unsafe_free()
        if rc != 0:
            raise Error("checkpoint: rename to " + self.cfg.path + " failed")
        self.last_ns = perf_counter_ns()
        print("Checkpoint " + self.cfg.path + " at " + String(done) + " (" + String(self.bytes // (1024 * 1024)) + " MB)")

    def _section(mut self, name: String, n: Int) raises:
        var got_name = self._str(name)
        var got_n = self._int(n)
        if got_name != name or got_n != n:
            raise Error("checkpoint " + self.cfg.resume + ": section '" + got_name + "' (" + String(got_n)
                        + " bytes) where this render expects '" + name + "' (" + String(n) + " bytes)")

    def io[T: AnyType](mut self, name: String, p: Pointer[T, MutUntrackedOrigin], n: Int) raises:
        """n host values, written or read back."""
        var nb = n * size_of[T]()
        self._section(name, nb)
        if self.writing:
            self._put(p.unsafe_bitcast[UInt8](), nb)
        else:
            self._get(p.unsafe_bitcast[UInt8](), nb)

    def io[dt: DType](mut self, name: String, ctx: DeviceContext, mut buf: DeviceBuffer[dt]) raises:
        """A whole device buffer, staged through host memory."""
        var n = len(buf)
        var h = unsafe_alloc[Scalar[dt]](max(n, 1))
        if self.writing:
            ctx.enqueue_copy(h, buf)
            ctx.synchronize()
        self.io(name, h, n)
        if not self.writing:
            ctx.enqueue_copy(buf, h)
            ctx.synchronize()
        h.unsafe_free()

    def flag(mut self, name: String, mut v: Bool) raises:
        var p = unsafe_alloc[Int32](1)
        p[unsafe_offset=0] = Int32(1) if v else Int32(0)
        self.io(name, p, 1)
        v = p[unsafe_offset=0] != Int32(0)
        p.unsafe_free()

    def scalar(mut self, name: String, mut v: Float32) raises:
        var p = unsafe_alloc[Float32](1)
        p[unsafe_offset=0] = v
        self.io(name, p, 1)
        v = p[unsafe_offset=0]
        p.unsafe_free()
