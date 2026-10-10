# Ptex demand paging. Every face starts at the scene's base resolution cap (pbrt_parser's
# _plan_ptex_budget); the renderer writes the resolution a lookup needed into the face table
# (shading._sample_ptex) and this cache loads those faces into one shared pool between sample
# batches, dropping the ones nobody used when the pool is full (second chance).
# Face table entry, 8 bytes: uint32 texel offset, log2 width, log2 height, wanted log2
# (255 = nothing finer exists), flags (1 = offset is into the pool, 2 = used).
from max.algorithm import parallelize
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores

comptime _Bytes = Pointer[UInt8, MutUntrackedOrigin]

@always_inline
def ptex_chain_texels(ul_in: Int, vl_in: Int) -> Int:
    """Texels in a face's mip chain from 2^ul x 2^vl down to 1x1, each level with its one-texel border."""
    var ul = ul_in; var vl = vl_in
    var n = 0
    while True:
        n += ((1 << ul) + 2) * ((1 << vl) + 2)
        if ul == 0 and vl == 0:
            return n
        ul = max(ul - 1, 0); vl = max(vl - 1, 0)

@always_inline
def _entry_off(e: _Bytes) -> Int:
    return Int(e.unsafe_bitcast[UInt32]()[unsafe_offset=0])

@always_inline
def _set_entry_off(e: _Bytes, off: Int):
    e.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(off)

struct PtexCache(Movable):
    var names: List[_Bytes]     # per file: its path (C string)
    var base: List[_Bytes]      # per file: the face table as packed
    var cur: List[_Bytes]       # per file: the face table as the renderer has it
    var n_faces: List[Int]
    var changed: List[Bool]
    var pool: _Bytes            # RGB8 texels
    var cap: Int                # pool size in texels
    var top: Int
    var dead: Int
    var chunks: List[Int]       # pool allocations in offset order: file, face (-1 = dead), offset, texels
    var requests: List[Int]     # pending: file, face, log2 width, log2 height, offset
    var loaded: Int
    var evicted: Int
    var denied: Int

    def __init__(out self, pool_bytes: Int):
        self.names = List[_Bytes](); self.base = List[_Bytes](); self.cur = List[_Bytes]()
        self.n_faces = List[Int](); self.changed = List[Bool]()
        self.cap = min(pool_bytes // 3, 0xFFFFFFFF)
        self.pool = unsafe_alloc[UInt8](max(self.cap * 3, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        self.top = 0; self.dead = 0
        self.chunks = List[Int](); self.requests = List[Int]()
        self.loaded = 0; self.evicted = 0; self.denied = 0

    def __del__(deinit self):
        for i in range(len(self.names)):
            self.names[i].unsafe_free(); self.base[i].unsafe_free(); self.cur[i].unsafe_free()
        self.pool.unsafe_free()

    def add_file(mut self, name: _Bytes, table: _Bytes, n_faces: Int) -> Int:
        var n = 0
        while name[unsafe_offset=n] != UInt8(0): n += 1
        var nm = unsafe_alloc[UInt8](n + 1).unsafe_origin_cast[MutUntrackedOrigin]()
        var b = unsafe_alloc[UInt8](n_faces * 8).unsafe_origin_cast[MutUntrackedOrigin]()
        var c = unsafe_alloc[UInt8](n_faces * 8).unsafe_origin_cast[MutUntrackedOrigin]()
        unsafe_memcpy(dest=nm, src=name, count=n + 1)
        unsafe_memcpy(dest=b, src=table, count=n_faces * 8)
        unsafe_memcpy(dest=c, src=table, count=n_faces * 8)
        self.names.append(nm); self.base.append(b); self.cur.append(c)
        self.n_faces.append(n_faces); self.changed.append(False)
        return len(self.names) - 1

    def feed(mut self, id: Int, dev: _Bytes):
        """Take one file's feedback: `dev` is its face table as the renderer left it."""
        var cur = self.cur[id]
        var tex = 0
        var res = unsafe_alloc[Int32](2)
        for f in range(self.n_faces[id]):
            var e = cur.unsafe_offset(f * 8)
            e[unsafe_offset=7] = e[unsafe_offset=7] | (dev[unsafe_offset=f * 8 + 7] & UInt8(2))
            var want = Int(dev[unsafe_offset=f * 8 + 6])
            if want == 0 or want == 255:
                continue
            if tex == 0:
                tex = external_call["ptex_open", Int, _Bytes](self.names[id])
                if tex == 0:
                    break
            _ = external_call["ptex_face_res", NoneType, Int, Int32, Pointer[Int32, MutUntrackedOrigin]](tex, Int32(f), res)
            var ul = min(Int(res[unsafe_offset=0]), want); var vl = min(Int(res[unsafe_offset=1]), want)
            self.changed[id] = True   # the renderer's wanted byte is reset either way
            if ul <= Int(e[unsafe_offset=4]) and vl <= Int(e[unsafe_offset=5]):
                e[unsafe_offset=6] = UInt8(255)
                continue
            self.requests.append(id); self.requests.append(f); self.requests.append(ul); self.requests.append(vl)
            self.requests.append(0)
        if tex != 0:
            _ = external_call["ptex_close", NoneType, Int](tex)
        res.unsafe_free()

    def _kill_chunk(mut self, off: Int):
        var lo = 0; var hi = len(self.chunks) // 4
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.chunks[mid * 4 + 2] < off:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.chunks) // 4 and self.chunks[lo * 4 + 2] == off and self.chunks[lo * 4 + 1] >= 0:
            self.chunks[lo * 4 + 1] = -1
            self.dead += self.chunks[lo * 4 + 3]

    def _evict_and_compact(mut self):
        """Drop the pool faces nobody used since the last call and close the holes."""
        var top = 0; var kept = 0
        for k in range(len(self.chunks) // 4):
            var file = self.chunks[k * 4]; var face = self.chunks[k * 4 + 1]
            var off = self.chunks[k * 4 + 2]; var texels = self.chunks[k * 4 + 3]
            if face < 0:
                continue
            var e = self.cur[file].unsafe_offset(face * 8)
            self.changed[file] = True
            if (Int(e[unsafe_offset=7]) & 2) == 0:
                unsafe_memcpy(dest=e, src=self.base[file].unsafe_offset(face * 8), count=8)
                self.evicted += 1
                continue
            e[unsafe_offset=7] = UInt8(1)
            if off != top:
                _ = external_call["memmove", _Bytes, _Bytes, _Bytes, Int](
                    self.pool.unsafe_offset(top * 3), self.pool.unsafe_offset(off * 3), texels * 3)
            _set_entry_off(e, top)
            self.chunks[kept * 4] = file; self.chunks[kept * 4 + 1] = face
            self.chunks[kept * 4 + 2] = top; self.chunks[kept * 4 + 3] = texels
            top += texels
            kept += 1
        self.chunks.resize(kept * 4, 0)
        self.top = top
        self.dead = 0

    def commit(mut self) -> Tuple[Int, Int, Int]:
        """Load what was asked for. Returns the number of faces loaded and the pool byte range to upload."""
        var top0 = self.top
        var n = 0; var denied = 0
        for r in range(len(self.requests) // 5):
            var file = self.requests[r * 5]; var face = self.requests[r * 5 + 1]
            var ul = self.requests[r * 5 + 2]; var vl = self.requests[r * 5 + 3]
            var e = self.cur[file].unsafe_offset(face * 8)
            var texels = ptex_chain_texels(ul, vl)
            e[unsafe_offset=6] = UInt8(0)
            if self.top + texels > self.cap:
                denied += 1
                continue
            if (Int(e[unsafe_offset=7]) & 1) != 0:
                self._kill_chunk(_entry_off(e))
            self.chunks.append(file); self.chunks.append(face); self.chunks.append(self.top); self.chunks.append(texels)
            self.requests[n * 5] = file; self.requests[n * 5 + 1] = face
            self.requests[n * 5 + 2] = ul; self.requests[n * 5 + 3] = vl; self.requests[n * 5 + 4] = self.top
            self.top += texels
            n += 1
        var reqs = self.requests.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        var names = self.names.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        var pool = self.pool
        var cursor = unsafe_alloc[Int32](1)
        cursor[unsafe_offset=0] = Int32(0)

        def read_worker(_worker_idx: Int) {imm}:
            var open_file = -1
            var tex = 0
            while True:
                var r0 = Int(Atomic.fetch_add(cursor, Int32(64)))
                if r0 >= n:
                    break
                for r in range(r0, min(r0 + 64, n)):
                    var file = reqs[unsafe_offset=r * 5]
                    if file != open_file:
                        if tex != 0:
                            _ = external_call["ptex_close", NoneType, Int](tex)
                        tex = external_call["ptex_open", Int, _Bytes](names[unsafe_offset=file])
                        open_file = file
                    if tex != 0:
                        _ = external_call["ptex_read_face", NoneType, Int, Int32, Int32, Int32, _Bytes](
                            tex, Int32(reqs[unsafe_offset=r * 5 + 1]), Int32(reqs[unsafe_offset=r * 5 + 2]),
                            Int32(reqs[unsafe_offset=r * 5 + 3]), pool.unsafe_offset(reqs[unsafe_offset=r * 5 + 4] * 3))
            if tex != 0:
                _ = external_call["ptex_close", NoneType, Int](tex)

        if n > 0:
            parallelize(read_worker, min(num_performance_cores(), n // 64 + 1))
        cursor.unsafe_free()
        for r in range(n):
            var e = self.cur[self.requests[r * 5]].unsafe_offset(self.requests[r * 5 + 1] * 8)
            _set_entry_off(e, self.requests[r * 5 + 4])
            e[unsafe_offset=4] = UInt8(self.requests[r * 5 + 2])
            e[unsafe_offset=5] = UInt8(self.requests[r * 5 + 3])
            e[unsafe_offset=7] = UInt8(3)
        self.requests.clear()
        self.loaded += n
        self.denied += denied
        if denied > 0:
            self._evict_and_compact()
            return (n, 0, self.top * 3)
        return (n, top0 * 3, self.top * 3)

    def take_table(mut self, id: Int, dst: _Bytes) -> Bool:
        """Copy file `id`'s face table to `dst` if it changed since the last call."""
        if not self.changed[id]:
            return False
        self.changed[id] = False
        unsafe_memcpy(dest=dst, src=self.cur[id], count=self.n_faces[id] * 8)
        return True
