from std.atomic import Atomic
from std.collections import Dict
from std.memory.alloc import unsafe_alloc
from std.os.path import getsize
from std.sys.info import num_performance_cores
from max.algorithm import parallelize
from .ply import load_ply

# The scene parse is sequential, but the PLY files it will ask for are independent of each other: a text scan of the
# scene (and its Include files) finds the plymesh filenames, loader threads load them while the parse runs, and the
# parse takes each mesh from here when its path matches (loading it itself if no loader got there first). A scan that
# misses or over-reports changes nothing but speed.

comptime _MAX_INCLUDE_DEPTH = 4
comptime _MAX_SCAN_BYTES = 64 * 1024 * 1024

struct PlyPrefetch(Movable):
    var slot: Dict[String, Int]
    var n: Int
    var uses: Pointer[Int32, MutUntrackedOrigin]      # scene references left; the mesh is freed at 0
    var ok: Pointer[Int32, MutUntrackedOrigin]
    var pts: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]
    var nv: Pointer[Int32, MutUntrackedOrigin]
    var idx: Pointer[Pointer[Int32, MutUntrackedOrigin], MutUntrackedOrigin]
    var nt: Pointer[Int32, MutUntrackedOrigin]
    var uvs: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]
    var has_uvs: Pointer[Int32, MutUntrackedOrigin]
    var nrm: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]
    var has_nrm: Pointer[Int32, MutUntrackedOrigin]
    var claim: Pointer[Int32, MutUntrackedOrigin]     # fetch_add == 0 wins the right to load slot i
    var done: Pointer[Int32, MutUntrackedOrigin]      # 1 once slot i's outputs are written
    var cursor: Pointer[Int32, MutUntrackedOrigin]    # next slot for a loader
    var cpaths: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]

    def __init__(out self):
        self.slot = Dict[String, Int]()
        self.n = 0
        self.uses = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.ok = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.pts = Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling()
        self.nv = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.idx = Pointer[Pointer[Int32, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling()
        self.nt = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.uvs = Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling()
        self.has_uvs = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.nrm = Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling()
        self.has_nrm = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.claim = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.done = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.cursor = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.cpaths = Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling()

    def _load_slot(self, k: Int):
        self.ok[unsafe_offset=k] = load_ply(self.cpaths[unsafe_offset=k], self.pts.unsafe_offset(k), self.nv.unsafe_offset(k),
            self.idx.unsafe_offset(k), self.nt.unsafe_offset(k), self.uvs.unsafe_offset(k), self.has_uvs.unsafe_offset(k),
            self.nrm.unsafe_offset(k), self.has_nrm.unsafe_offset(k), quiet=True)
        Atomic.store(self.done.unsafe_offset(k), Int32(1))

    def run_loader(self):
        """Load slots until none is left unclaimed (called by loader threads)."""
        while True:
            var k = Int(Atomic.fetch_add(self.cursor, Int32(1)))
            if k >= self.n:
                return
            if Atomic.fetch_add(self.claim.unsafe_offset(k), Int32(1)) == Int32(0):
                self._load_slot(k)

    def _ensure(self, k: Int):
        if Atomic.fetch_add(self.claim.unsafe_offset(k), Int32(1)) == Int32(0):
            self._load_slot(k)
        else:
            while Atomic.load(self.done.unsafe_offset(k)) == Int32(0):
                pass

    def lookup(self, path: String) -> Int:
        """Slot of a successfully prefetched mesh with a reference left, else -1."""
        if self.n == 0:
            return -1
        var i = self.slot.get(path, -1)
        if i < 0 or self.uses[unsafe_offset=i] <= Int32(0):
            return -1
        self._ensure(i)
        if self.ok[unsafe_offset=i] == Int32(0):
            return -1
        return i

    def _free_slot(self, i: Int):
        if self.ok[unsafe_offset=i] == Int32(0):
            return
        self.ok[unsafe_offset=i] = Int32(0)
        self.pts[unsafe_offset=i].unsafe_free()
        self.idx[unsafe_offset=i].unsafe_free()
        if self.has_uvs[unsafe_offset=i] != Int32(0):
            self.uvs[unsafe_offset=i].unsafe_free()
        if self.has_nrm[unsafe_offset=i] != Int32(0):
            self.nrm[unsafe_offset=i].unsafe_free()

    def release(self, i: Int):
        """One reference consumed; the last one frees the mesh."""
        self.uses[unsafe_offset=i] -= Int32(1)
        if self.uses[unsafe_offset=i] <= Int32(0):
            self._free_slot(i)

    def free_all(mut self):
        if self.n == 0:
            return
        for i in range(self.n):
            if Atomic.fetch_add(self.claim.unsafe_offset(i), Int32(1)) != Int32(0):   # loaded or loading: wait, then free
                while Atomic.load(self.done.unsafe_offset(i)) == Int32(0):
                    pass
                self._free_slot(i)
            self.cpaths[unsafe_offset=i].unsafe_free()
        self.claim.unsafe_free(); self.done.unsafe_free(); self.cursor.unsafe_free(); self.cpaths.unsafe_free()
        self.uses.unsafe_free(); self.ok.unsafe_free(); self.pts.unsafe_free(); self.nv.unsafe_free()
        self.idx.unsafe_free(); self.nt.unsafe_free(); self.uvs.unsafe_free(); self.has_uvs.unsafe_free()
        self.nrm.unsafe_free(); self.has_nrm.unsafe_free()
        self.n = 0

def _tok_is(text: Pointer[UInt8, MutUntrackedOrigin], a: Int, b: Int, lit: StringLiteral) -> Bool:
    var lp = lit.ptr()
    var n = 0
    while lp[unsafe_offset=n] != UInt8(0):
        n += 1
    if b - a != n:
        return False
    for k in range(n):
        if text[unsafe_offset=a + k] != lp[unsafe_offset=k]:
            return False
    return True

def _tok_ends(text: Pointer[UInt8, MutUntrackedOrigin], a: Int, b: Int, lit: StringLiteral) -> Bool:
    var lp = lit.ptr()
    var n = 0
    while lp[unsafe_offset=n] != UInt8(0):
        n += 1
    if b - a < n:
        return False
    for k in range(n):
        if text[unsafe_offset=b - n + k] != lp[unsafe_offset=k]:
            return False
    return True

def _make_string(text: Pointer[UInt8, MutUntrackedOrigin], a: Int, b: Int) -> String:
    var tmp = unsafe_alloc[UInt8](b - a + 1)
    for k in range(b - a):
        tmp[unsafe_offset=k] = text[unsafe_offset=a + k]
    tmp[unsafe_offset=b - a] = UInt8(0)
    var s = String(unsafe_from_utf8_ptr=tmp.as_imm())
    tmp.unsafe_free()
    return s^

def _resolve(scene_dir: String, name: String) -> String:
    return name if name.startswith("/") else scene_dir + name

def _add_path(path: String, mut paths: List[String], mut slot: Dict[String, Int], mut uses: List[Int32]):
    if path.endswith(".gz"):
        return
    var i = slot.get(path, -1)
    if i >= 0:
        uses[i] += Int32(1)
    else:
        slot[path] = len(paths)
        paths.append(path)
        uses.append(Int32(1))

def _scan_filename_after(text: Pointer[UInt8, MutUntrackedOrigin], size: Int, start: Int, scene_dir: String,
                         mut paths: List[String], mut slot: Dict[String, Int], mut uses: List[Int32]):
    # The parameter list of one Shape: quoted tokens up to the next directive keyword (an unquoted capital).
    var i = start
    var limit = min(size, start + 8192)
    while i < limit:
        var c = text[unsafe_offset=i]
        if c >= UInt8(65) and c <= UInt8(90):
            return
        if c == UInt8(35):
            while i < limit and text[unsafe_offset=i] != UInt8(10):
                i += 1
            continue
        if c == UInt8(34):
            var j = i + 1
            while j < size and text[unsafe_offset=j] != UInt8(34):
                j += 1
            if _tok_ends(text, i + 1, j, "filename"):
                var v0 = j + 1
                while v0 < size and text[unsafe_offset=v0] != UInt8(34):
                    v0 += 1
                var v1 = v0 + 1
                while v1 < size and text[unsafe_offset=v1] != UInt8(34):
                    v1 += 1
                if v1 < size and v1 - v0 - 1 > 0:
                    _add_path(_resolve(scene_dir, _make_string(text, v0 + 1, v1)), paths, slot, uses)
                return
            i = j + 1
            continue
        i += 1

def _scan_scene_text(text: Pointer[UInt8, MutUntrackedOrigin], size: Int, scene_dir: String, depth: Int,
                     mut paths: List[String], mut slot: Dict[String, Int], mut uses: List[Int32]):
    var i = 0
    while i < size:
        var c = text[unsafe_offset=i]
        if c == UInt8(35):
            while i < size and text[unsafe_offset=i] != UInt8(10):
                i += 1
            continue
        if c == UInt8(34):
            var j = i + 1
            while j < size and text[unsafe_offset=j] != UInt8(34):
                j += 1
            if _tok_is(text, i + 1, j, "plymesh"):
                _scan_filename_after(text, size, j + 1, scene_dir, paths, slot, uses)
            i = j + 1
            continue
        if c == UInt8(73) and depth < _MAX_INCLUDE_DEPTH and i + 8 < size:     # 'I'nclude / 'I'mport
            var w_end = i
            while w_end < size and text[unsafe_offset=w_end] > UInt8(32):
                w_end += 1
            if _tok_is(text, i, w_end, "Include") or _tok_is(text, i, w_end, "Import"):
                var q0 = w_end
                while q0 < size and text[unsafe_offset=q0] <= UInt8(32):
                    q0 += 1
                if q0 < size and text[unsafe_offset=q0] == UInt8(34):
                    var q1 = q0 + 1
                    while q1 < size and text[unsafe_offset=q1] != UInt8(34):
                        q1 += 1
                    var inc = _resolve(scene_dir, _make_string(text, q0 + 1, q1))
                    if not inc.endswith(".gz"):
                        try:
                            if getsize(inc) > _MAX_SCAN_BYTES:    # geometry dumps (curves, millions of triangles) name no meshes
                                i = q1 + 1
                                continue
                            var bytes = open(inc, "r").read_bytes()
                            _scan_scene_text(bytes.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                                             len(bytes), scene_dir, depth + 1, paths, slot, uses)
                        except:
                            pass
                    i = q1 + 1
                    continue
            i = w_end
            continue
        i += 1

def scan_plys(text: Pointer[UInt8, MutUntrackedOrigin], size: Int, scene_dir: String) -> PlyPrefetch:
    """Scan the scene text for plymesh files; the meshes are loaded by run_loader / lookup."""
    var paths = List[String]()
    var slot = Dict[String, Int]()
    var uses = List[Int32]()
    _scan_scene_text(text, size, scene_dir, 0, paths, slot, uses)
    var pf = PlyPrefetch()
    var n = len(paths)
    if n == 0:
        return pf^
    pf.n = n
    pf.slot = slot^
    pf.uses = unsafe_alloc[Int32](n)
    pf.ok = unsafe_alloc[Int32](n)
    pf.pts = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](n)
    pf.nv = unsafe_alloc[Int32](n)
    pf.idx = unsafe_alloc[Pointer[Int32, MutUntrackedOrigin]](n)
    pf.nt = unsafe_alloc[Int32](n)
    pf.uvs = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](n)
    pf.has_uvs = unsafe_alloc[Int32](n)
    pf.nrm = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](n)
    pf.has_nrm = unsafe_alloc[Int32](n)
    pf.claim = unsafe_alloc[Int32](n)
    pf.done = unsafe_alloc[Int32](n)
    pf.cursor = unsafe_alloc[Int32](1)
    pf.cursor[unsafe_offset=0] = Int32(0)
    pf.cpaths = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](n)
    for i in range(n):
        pf.uses[unsafe_offset=i] = uses[i]
        pf.ok[unsafe_offset=i] = Int32(0)
        pf.claim[unsafe_offset=i] = Int32(0)
        pf.done[unsafe_offset=i] = Int32(0)
        var pl = paths[i].byte_length()
        var cp = unsafe_alloc[UInt8](pl + 1)
        for k in range(pl):
            cp[unsafe_offset=k] = paths[i].as_bytes()[k]
        cp[unsafe_offset=pl] = UInt8(0)
        pf.cpaths[unsafe_offset=i] = cp
    return pf^

def load_all_plys(pf: PlyPrefetch):
    """Load every slot on all cores and return when done (for callers with no loader threads of their own)."""
    if pf.n == 0:
        return
    def worker(_w: Int) {imm pf}:
        pf.run_loader()
    parallelize(worker, min(num_performance_cores(), pf.n))
