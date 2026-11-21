# Tiled Ptex demand paging: levels finer than a face's resident tail come as 64x64 tiles (oiio.cc
# ptex_pack_tiled, texturing._ptex_tiled_level). The GPU appends the pages it asks for to a list in the pool
# header; between sample batches this cache reads the list, gives each page a free slot, loads the tiles on
# threads of its own and, once loaded, has them copied into their slots and their page entries set. Slots no
# lookup used since the last sweep are freed when free slots run short (a clock over the used bytes).
from gonzales.core.os_thread import OsThread, ThreadArg
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores
from std.builtin.sort import sort

comptime _Bytes = Pointer[UInt8, MutUntrackedOrigin]
# Layout shared with texturing._ptex_tiled_level: a page entry is 0 (absent), 1 (asked for), 2 (no room),
# 3 (loading) or 16 + the slot. Pool header byte 0 switches withholding, byte 2 picks its policy (TILE_WITHHOLD_*).
comptime TILE_WITHHOLD_ALL = 0       # every lookup that misses its tile
comptime TILE_WITHHOLD_LOADING = 1   # only tiles already loading
comptime TILE_WITHHOLD_LOADING2 = 2  # only tiles loading and two or more levels finer than the resident one
comptime TILE_WITHHOLD_CAMERA = 3    # only camera hits
comptime PTEX_TILE_LOG2 = 6
comptime PTEX_SLOT_W = 66
comptime PTEX_SLOT_BYTES = PTEX_SLOT_W * PTEX_SLOT_W * 3
comptime TILE_REQ_COUNT = 8      # pool header bytes: uint32 requests made, uint32 capacity, uint32 slots,
comptime TILE_REQ_CAP = 12       # uint64 byte offset of the used bytes, uint64 byte offset of the slots
comptime TILE_SLOTS = 16
comptime TILE_USED_OFF = 24
comptime TILE_SLOTS_OFF = 32
comptime TILE_REQS = 64          # requests: (file, page entry byte offset in its blob) pairs of uint32

@always_inline
def _tiles_at(ul: Int, vl: Int) -> Int:
    return (1 << max(ul - PTEX_TILE_LOG2, 0)) * (1 << max(vl - PTEX_TILE_LOG2, 0))

comptime TILE_REQ_MAX = 1 << 20    # requests a batch can make before the list is full
comptime TILE_LOAD_MAX = 65536      # tiles a load takes (856 MB of host staging); the rest wait in the backlog
comptime TILE_STAGE_CHUNK = 16384   # tiles a device staging chunk holds (213 MB)
comptime PAGE_ABSENT = 0
comptime PAGE_NO_ROOM = 2
comptime PAGE_LOADING = 3       # has a slot, its tile comes with the next finish()
comptime PAGE_SLOT = 16


@fieldwise_init
struct _TileJob(TrivialRegisterPassable):
    """What the loader threads share: groups of tiles of one face level, and where to put each tile."""
    var groups: Pointer[Int, MutUntrackedOrigin]   # per group: file, face, log2 width, log2 height, first tile, tile count
    var n_groups: Int
    var tiles: Pointer[Int, MutUntrackedOrigin]    # per tile: tx, ty, staging index
    var names: Pointer[_Bytes, MutUntrackedOrigin]
    var staging: _Bytes
    var cursor: Pointer[Int32, MutUntrackedOrigin]
    var done: Pointer[Int32, MutUntrackedOrigin]


def _tile_load_main(arg: ThreadArg) -> ThreadArg:
    var job = arg.unsafe_bitcast[_TileJob]()[unsafe_offset=0]
    var open_file = -1
    var tex = 0
    var level = List[UInt8]()
    while True:
        var g = Int(Atomic.fetch_add(job.cursor, Int32(1)))
        if g >= job.n_groups:
            break
        var file = job.groups[unsafe_offset=g * 6]
        if file != open_file:
            if tex != 0:
                _ = external_call["ptex_close", NoneType, Int](tex)
            tex = external_call["ptex_open", Int, _Bytes](job.names[unsafe_offset=file])
            open_file = file
        if tex == 0:
            continue
        var ul = job.groups[unsafe_offset=g * 6 + 2]; var vl = job.groups[unsafe_offset=g * 6 + 3]
        level.resize(((1 << ul) + 2) * ((1 << vl) + 2) * 3, UInt8(0))
        var lp = level.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        _ = external_call["ptex_read_level", NoneType, Int, Int32, Int32, Int32, _Bytes](
            tex, Int32(job.groups[unsafe_offset=g * 6 + 1]), Int32(ul), Int32(vl), lp)
        var t0 = job.groups[unsafe_offset=g * 6 + 4]
        for t in range(t0, t0 + job.groups[unsafe_offset=g * 6 + 5]):
            _ = external_call["ptex_crop_tile", NoneType, _Bytes, Int32, Int32, Int32, Int32, _Bytes](
                lp, Int32(ul), Int32(vl), Int32(job.tiles[unsafe_offset=t * 3]), Int32(job.tiles[unsafe_offset=t * 3 + 1]),
                job.staging.unsafe_offset(job.tiles[unsafe_offset=t * 3 + 2] * PTEX_SLOT_BYTES))
    if tex != 0:
        _ = external_call["ptex_close", NoneType, Int](tex)
    _ = Atomic.fetch_add(job.done, Int32(1))
    return arg


struct TileCache(Movable):
    var names: List[_Bytes]      # per file: its path (C string)
    var info: List[_Bytes]       # per file: its tile entries (page table byte offset, full log2 w, h, resident level)
    var n_faces: List[Int]
    var pages_lo: List[Int]      # per file: the byte range of its page tables
    var pages_hi: List[Int]
    var rejected: Int            # requests outside their file's page tables (a lookup bug; GONZALES_GPU_LOG)
    var pool_bytes: Int
    var req_cap: Int
    var used_off: Int
    var slots_off: Int
    var n_slots: Int
    var free: List[Int32]        # free slots
    var owner: List[Int]         # per slot: file << 32 | page entry byte offset, -1 free
    var fresh: List[Bool]        # per slot: loaded since the last sweep, so no lookup could use it yet
    var loads: List[Int]         # the load in flight: file, entry byte, slot
    var no_room: List[Int]       # file << 32 | entry byte of pages marked no room, asked again after a sweep
    var backlog: List[Int]       # asked for but beyond the last load: still marked asked, placed by a later begin()
    var patches: List[UInt32]    # page entries to write: file, entry byte, value
    var staging: _Bytes
    var staging_cap: Int         # tiles
    var loaded: Int
    var evicted: Int
    var denied: Int
    var loading: Bool
    var asked: Int               # requests the last begin() read, and the distinct pages among them
    var asked_pages: Int
    var job: Pointer[_TileJob, MutUntrackedOrigin]
    var groups: List[Int]
    var tiles: List[Int]
    var threads: List[OsThread]

    def __init__(out self, pool_bytes: Int):
        self.names = List[_Bytes](); self.info = List[_Bytes](); self.n_faces = List[Int]()
        self.pages_lo = List[Int](); self.pages_hi = List[Int](); self.rejected = 0
        self.pool_bytes = 0; self.req_cap = TILE_REQ_MAX; self.used_off = 0; self.slots_off = 0; self.n_slots = 0
        self.free = List[Int32](); self.owner = List[Int](); self.fresh = List[Bool]()
        self.loads = List[Int](); self.no_room = List[Int](); self.patches = List[UInt32]()
        self.backlog = List[Int]()
        self.staging_cap = 0; self.staging = _Bytes.unsafe_dangling()
        self.loaded = 0; self.evicted = 0; self.denied = 0; self.loading = False
        self.asked = 0; self.asked_pages = 0
        self.job = unsafe_alloc[_TileJob](1).unsafe_origin_cast[MutUntrackedOrigin]()
        self.job[unsafe_offset=0] = _TileJob(Pointer[Int, MutUntrackedOrigin].unsafe_dangling(), 0,
            Pointer[Int, MutUntrackedOrigin].unsafe_dangling(), Pointer[_Bytes, MutUntrackedOrigin].unsafe_dangling(), self.staging,
            unsafe_alloc[Int32](1).unsafe_origin_cast[MutUntrackedOrigin](), unsafe_alloc[Int32](1).unsafe_origin_cast[MutUntrackedOrigin]())
        self.groups = List[Int](); self.tiles = List[Int](); self.threads = List[OsThread]()
        self.set_pool(pool_bytes)

    def set_pool(mut self, pool_bytes: Int):
        """Lay the pool out: header, request list, a used byte per slot, the slots. All slots free."""
        self.pool_bytes = pool_bytes
        self.used_off = (TILE_REQS + self.req_cap * 8 + 63) // 64 * 64
        var rest = max(pool_bytes - self.used_off - 64, 0)
        self.n_slots = rest // (PTEX_SLOT_BYTES + 1)
        self.slots_off = (self.used_off + self.n_slots + 63) // 64 * 64
        self.free = List[Int32](capacity=self.n_slots)
        for s in range(self.n_slots - 1, -1, -1): self.free.append(Int32(s))
        self.owner = List[Int](length=self.n_slots, fill=-1)
        self.fresh = List[Bool](length=self.n_slots, fill=False)

    def __deinit__(deinit self):
        for t in range(len(self.threads)): self.threads[t].join()
        for i in range(len(self.names)):
            self.names[i].unsafe_free(); self.info[i].unsafe_free()
        if self.staging_cap > 0:
            self.staging.unsafe_free()
        self.job[unsafe_offset=0].cursor.unsafe_free(); self.job[unsafe_offset=0].done.unsafe_free()
        self.job.unsafe_free()

    def header(self) -> List[UInt8]:
        """The pool header the device starts with: no requests, the list capacity, the slot layout."""
        var h = List[UInt8](length=TILE_REQS, fill=UInt8(0))
        var p = h.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        p.unsafe_offset(TILE_REQ_CAP).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(self.req_cap)
        p.unsafe_offset(TILE_SLOTS).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(self.n_slots)
        p.unsafe_offset(TILE_USED_OFF).unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(self.used_off)
        p.unsafe_offset(TILE_SLOTS_OFF).unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(self.slots_off)
        return h^

    def add_file(mut self, name: _Bytes, blob: _Bytes, n_faces: Int, n_bytes: Int = 0) -> Int:
        """A tiled file: `blob` is its packed form (face entries, tile entries, texels, page tables), n_bytes long."""
        var n = 0
        while name[unsafe_offset=n] != UInt8(0): n += 1
        var nm = unsafe_alloc[UInt8](n + 1).unsafe_origin_cast[MutUntrackedOrigin]()
        unsafe_memcpy(dest=nm, src=name, count=n + 1)
        var inf = unsafe_alloc[UInt8](n_faces * 8).unsafe_origin_cast[MutUntrackedOrigin]()
        unsafe_memcpy(dest=inf, src=blob.unsafe_offset(n_faces * 8), count=n_faces * 8)
        self.names.append(nm); self.info.append(inf); self.n_faces.append(n_faces)
        self.pages_lo.append(Int(inf.unsafe_bitcast[UInt32]()[unsafe_offset=0]) if n_faces > 0 else 0)
        self.pages_hi.append(n_bytes if n_bytes > 0 else 1 << 62)
        return len(self.names) - 1

    def _decode(self, file: Int, entry: Int) -> Tuple[Int, Int, Int, Int, Int]:
        """(face, log2 width, log2 height, tx, ty) of the page whose entry sits at byte `entry` of the file's blob."""
        var inf = self.info[file]
        var lo = 0; var hi = self.n_faces[file] - 1
        while lo < hi:   # the last face whose page table starts at or before `entry`
            var mid = (lo + hi + 1) // 2
            if Int(inf.unsafe_offset(mid * 8).unsafe_bitcast[UInt32]()[unsafe_offset=0]) <= entry:
                lo = mid
            else:
                hi = mid - 1
        var e = inf.unsafe_offset(lo * 8)
        var idx = (entry - Int(e.unsafe_bitcast[UInt32]()[unsafe_offset=0])) // 4
        var fu = Int(e[unsafe_offset=4]); var fv = Int(e[unsafe_offset=5])
        var l = 0
        while True:
            var n = _tiles_at(max(fu - l, 0), max(fv - l, 0))
            if idx < n:
                break
            idx -= n
            l += 1
        var ul = max(fu - l, 0); var vl = max(fv - l, 0)
        var ntx = 1 << max(ul - PTEX_TILE_LOG2, 0)
        return (lo, ul, vl, idx % ntx, idx // ntx)

    def begin(mut self, reqs: Pointer[UInt32, MutUntrackedOrigin], n_req: Int) -> Int:
        """Take the requests read back from the device, give each new page a slot and start loading.
        Pages that find no slot are marked no room. Returns the number of tiles being loaded."""
        # A page is asked for once while marked so (texturing._tile_ask), so the requests hold no duplicates.
        var keys = self.backlog^   # the oldest first
        self.backlog = List[Int]()
        var n_old = len(keys)
        for i in range(n_req):
            var file = Int(reqs[unsafe_offset=2 * i]); var entry = Int(reqs[unsafe_offset=2 * i + 1])
            if file >= len(self.names):
                continue
            if entry < self.pages_lo[file] or entry + 4 > self.pages_hi[file] or (entry & 3) != 0:
                if self.rejected < 8:
                    print("tiles: rejected request file", file, "entry", entry, "pages", self.pages_lo[file], "..", self.pages_hi[file])
                self.rejected += 1
                continue
            keys.append((file << 32) | entry)
        self.asked = n_req; self.asked_pages = len(keys)
        # Group by file, face and level so each level is read once: the decoded tuple sorts that way.
        var recs = List[Int]()   # file, face, ul, vl, tx, ty, entry
        var avail = len(self.free)
        var take = TILE_LOAD_MAX
        for ki in range(len(keys)):
            var k = keys[ki]
            var file = k >> 32; var entry = k & 0xFFFFFFFF
            if take == 0 or (avail == 0 and ki < n_old):   # it stays asked for and waits for a later load
                self.backlog.append(k)
                continue
            if avail == 0:
                self.patches.append(UInt32(file)); self.patches.append(UInt32(entry)); self.patches.append(UInt32(PAGE_NO_ROOM))
                self.no_room.append(k)
                self.denied += 1
                continue
            avail -= 1
            take -= 1
            var (face, ul, vl, tx, ty) = self._decode(file, entry)
            recs.append(file); recs.append(face); recs.append(ul); recs.append(vl); recs.append(tx); recs.append(ty)
            recs.append(entry)
        var n = len(recs) // 7
        # Sort by (file, face, level): packed keys, the request index in the low 20 bits (TILE_REQ_MAX).
        var order = List[Int](capacity=n)
        for i in range(n):
            order.append((recs[i * 7] << 52) | (recs[i * 7 + 1] << 28) | (recs[i * 7 + 2] << 24) | (recs[i * 7 + 3] << 20) | i)
        sort(Span(order))
        for j in range(n): order[j] = order[j] & 0xFFFFF
        if n > self.staging_cap:
            if self.staging_cap > 0:
                self.staging.unsafe_free()
            self.staging_cap = n
            self.staging = unsafe_alloc[UInt8](n * PTEX_SLOT_BYTES).unsafe_origin_cast[MutUntrackedOrigin]()
        self.groups.clear(); self.tiles.clear(); self.loads.clear()
        for j in range(n):
            var i = order[j]
            var slot = Int(self.free.pop())
            var file = recs[i * 7]; var entry = recs[i * 7 + 6]
            self.owner[slot] = (file << 32) | entry
            self.loads.append(file); self.loads.append(entry); self.loads.append(slot)
            self.patches.append(UInt32(file)); self.patches.append(UInt32(entry)); self.patches.append(UInt32(PAGE_LOADING))
            var new_group = len(self.groups) == 0
            if not new_group:
                var g = len(self.groups) - 6
                for c in range(4):
                    if self.groups[g + c] != recs[i * 7 + c]:
                        new_group = True
            if new_group:
                for c in range(4): self.groups.append(recs[i * 7 + c])
                self.groups.append(j); self.groups.append(0)
            self.groups[len(self.groups) - 1] += 1
            self.tiles.append(recs[i * 7 + 4]); self.tiles.append(recs[i * 7 + 5]); self.tiles.append(j)
        self.loading = True
        if n == 0:
            return 0
        self.job[unsafe_offset=0] = _TileJob(
            self.groups.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), len(self.groups) // 6,
            self.tiles.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            self.names.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), self.staging,
            self.job[unsafe_offset=0].cursor, self.job[unsafe_offset=0].done)
        self.job[unsafe_offset=0].cursor[unsafe_offset=0] = Int32(0)
        self.job[unsafe_offset=0].done[unsafe_offset=0] = Int32(0)
        var n_threads = min(num_performance_cores(), len(self.groups) // 6)
        for t in range(n_threads):
            self.threads.append(OsThread())
            self.threads[t].start(_tile_load_main, self.job.unsafe_bitcast[UInt8]())
        return n

    def ready(self) -> Bool:
        return self.loading and Int(self.job[unsafe_offset=0].done[unsafe_offset=0]) >= len(self.threads)

    def finish(mut self) -> Int:
        """Wait for the load; its tiles are in `staging` in the order of `loads`. Returns how many."""
        for t in range(len(self.threads)): self.threads[t].join()
        self.threads.clear()
        self.loading = False
        var n = len(self.loads) // 3
        for i in range(n):
            self.patches.append(UInt32(self.loads[i * 3])); self.patches.append(UInt32(self.loads[i * 3 + 1]))
            self.patches.append(UInt32(PAGE_SLOT + self.loads[i * 3 + 2]))
            self.fresh[self.loads[i * 3 + 2]] = True
        self.loaded += n
        return n

    def sweep(mut self, used: _Bytes) -> Int:
        """Free the slots no lookup used since the last sweep (`used` holds a byte per slot) and let the
        pages that found no room ask again. Returns how many slots were freed."""
        var freed = 0
        for s in range(self.n_slots):
            if self.fresh[s]:
                self.fresh[s] = False
                continue
            if self.owner[s] < 0 or used[unsafe_offset=s] != UInt8(0):
                continue
            var k = self.owner[s]
            self.patches.append(UInt32(k >> 32)); self.patches.append(UInt32(k & 0xFFFFFFFF)); self.patches.append(UInt32(PAGE_ABSENT))
            self.owner[s] = -1
            self.free.append(Int32(s))
            freed += 1
        self.evicted += freed
        # As many pages that found no room as could now get a slot, twice over, ask again; the oldest first.
        var n_again = min(len(self.no_room), 2 * freed)
        for i in range(n_again):
            var k = self.no_room[i]
            self.patches.append(UInt32(k >> 32)); self.patches.append(UInt32(k & 0xFFFFFFFF)); self.patches.append(UInt32(PAGE_ABSENT))
        if n_again > 0:
            var rest = List[Int](capacity=len(self.no_room) - n_again)
            for i in range(n_again, len(self.no_room)): rest.append(self.no_room[i])
            self.no_room = rest^
        return freed
