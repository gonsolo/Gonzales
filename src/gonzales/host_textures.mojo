from .materials import Material
from .render_state import GpuTexture
from .os_thread import OsThread, ThreadArg
from max.algorithm import parallelize
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores

# Host-side decode of a scene's image textures into mip pyramids: the source of
# the GPU upload (gpu_scene.mojo) and of the CPU renderer's in-memory table.

def _cstr_eq(a: Pointer[UInt8, MutUntrackedOrigin], b: Pointer[UInt8, MutUntrackedOrigin]) -> Bool:
    var i = 0
    while True:
        var ca = a[unsafe_offset=i]
        var cb = b[unsafe_offset=i]
        if ca != cb:
            return False
        if ca == UInt8(0):
            return True
        i += 1

# (levels, total texels) of a full mip pyramid down to 1x1.
def _mip_texel_count(tw: Int, th: Int) -> Tuple[Int, Int]:
    var nlev = 1; var texels = tw * th
    var ww = tw; var hh = th
    while ww > 1 or hh > 1:
        ww = max(1, ww // 2); hh = max(1, hh // 2)
        nlev += 1; texels += ww * hh
    return (nlev, texels)

# The byte whose decoded value in `lut` (256 entries, non-decreasing) is nearest `v`.
@always_inline
def _nearest_lut_byte(lut: Pointer[Float32, MutUntrackedOrigin], v: Float32) -> UInt8:
    var lo = 0; var hi = 255
    while lo < hi:
        var mid = (lo + hi) // 2
        if lut[unsafe_offset=mid] < v:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and v - lut[unsafe_offset=lo - 1] <= lut[unsafe_offset=lo] - v:
        return UInt8(lo - 1)
    return UInt8(lo)

comptime _INV_LUT_SIZE: Int = 65536

@always_inline
def _dist(a: Float32, b: Float32) -> Float32:
    return a - b if a > b else b - a

# inv[q] = _nearest_lut_byte(lut, q / (_INV_LUT_SIZE - 1)): a candidate byte for
# each of _INV_LUT_SIZE evenly spaced values in [0, 1].
def _build_inverse_lut(lut: Pointer[Float32, MutUntrackedOrigin], inv: Pointer[UInt8, MutUntrackedOrigin]):
    for q in range(_INV_LUT_SIZE):
        inv[unsafe_offset=q] = _nearest_lut_byte(lut, Float32(q) / Float32(_INV_LUT_SIZE - 1))

# Same byte as _nearest_lut_byte(lut, v), in O(1). The grid is far finer than
# the LUT spacing, so the candidate is the nearest byte or a neighbour of it;
# |lut[b] - v| is unimodal in b, so walking up while strictly closer and down
# while no farther lands on the nearest byte, ties going to the lower one.
@always_inline
def _quantize_to_lut_byte(
    lut: Pointer[Float32, MutUntrackedOrigin], inv: Pointer[UInt8, MutUntrackedOrigin], v: Float32,
) -> UInt8:
    var q = Int(v * Float32(_INV_LUT_SIZE - 1) + Float32(0.5))
    if q < 0: q = 0
    if q > _INV_LUT_SIZE - 1: q = _INV_LUT_SIZE - 1
    var b = Int(inv[unsafe_offset=q])
    while b < 255 and _dist(lut[unsafe_offset=b + 1], v) < _dist(lut[unsafe_offset=b], v):
        b += 1
    while b > 0 and _dist(lut[unsafe_offset=b - 1], v) <= _dist(lut[unsafe_offset=b], v):
        b -= 1
    return UInt8(b)

# Fill `pyr` with a uint8 mip pyramid of `src` (tw x th, c channels). Each coarser
# level is the 2x2 box average in LINEAR space (bytes decoded through `lut`) --
# the same averages the float path computes -- stored as the nearest byte (via
# `inv`, the matching _build_inverse_lut table). The averages are carried in
# float between levels so rounding doesn't compound.
def _fill_u8_mips(
    pyr: Pointer[UInt8, MutUntrackedOrigin], src: Pointer[UInt8, MutUntrackedOrigin],
    tw: Int, th: Int, c: Int, lut: Pointer[Float32, MutUntrackedOrigin],
    inv: Pointer[UInt8, MutUntrackedOrigin],
):
    unsafe_memcpy(dest=pyr, src=src, count=tw * th * c)
    var prev = unsafe_alloc[Float32](tw * th * c)
    for i in range(tw * th * c):
        prev[unsafe_offset=i] = lut[unsafe_offset=Int(src[unsafe_offset=i])]
    var cur = unsafe_alloc[Float32](max(1, tw // 2) * max(1, th // 2) * c)
    var off_cur = tw * th * c
    var pw = tw; var ph = th
    while pw > 1 or ph > 1:
        var cw = max(1, pw // 2); var ch = max(1, ph // 2)
        for y in range(ch):
            for x in range(cw):
                var x0 = 2 * x; var x1 = min(2 * x + 1, pw - 1)
                var y0 = 2 * y; var y1 = min(2 * y + 1, ph - 1)
                for k in range(c):
                    var avg = (prev[unsafe_offset=(y0 * pw + x0) * c + k] + prev[unsafe_offset=(y0 * pw + x1) * c + k]
                               + prev[unsafe_offset=(y1 * pw + x0) * c + k] + prev[unsafe_offset=(y1 * pw + x1) * c + k]) * Float32(0.25)
                    cur[unsafe_offset=(y * cw + x) * c + k] = avg
                    pyr[unsafe_offset=off_cur + (y * cw + x) * c + k] = _quantize_to_lut_byte(lut, inv, avg)
        off_cur += cw * ch * c
        var tmp = prev; prev = cur; cur = tmp
        pw = cw; ph = ch
    prev.unsafe_free(); cur.unsafe_free()

# Fill `pyr` with a Float32 RGB mip pyramid of `src` (tw x th, linear RGB): level 0
# copied, each coarser level the 2x2 box average of the one before -- the float
# twin of _fill_u8_mips.
def _fill_f32_mips(
    pyr: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin],
    tw: Int, th: Int,
):
    unsafe_memcpy(dest=pyr, src=src, count=tw * th * 3)
    var off_prev = 0; var off_cur = tw * th * 3
    var pw = tw; var ph = th
    while pw > 1 or ph > 1:
        var cw = max(1, pw // 2); var ch = max(1, ph // 2)
        for y in range(ch):
            for x in range(cw):
                var x0 = 2 * x; var x1 = min(2 * x + 1, pw - 1)
                var y0 = 2 * y; var y1 = min(2 * y + 1, ph - 1)
                for k in range(3):
                    var a = pyr[unsafe_offset=off_prev + (y0 * pw + x0) * 3 + k]
                    var b = pyr[unsafe_offset=off_prev + (y0 * pw + x1) * 3 + k]
                    var cc = pyr[unsafe_offset=off_prev + (y1 * pw + x0) * 3 + k]
                    var d = pyr[unsafe_offset=off_prev + (y1 * pw + x1) * 3 + k]
                    pyr[unsafe_offset=off_cur + (y * cw + x) * 3 + k] = (a + b + cc + d) * Float32(0.25)
        off_prev = off_cur; off_cur += cw * ch * 3
        pw = cw; ph = ch

@fieldwise_init
struct _HostTexture(TrivialRegisterPassable):
    """One texture's mip pyramid, decoded on the host and ready to upload:
    `n_bytes` bytes at `data` -- UInt8 texels for FORMAT_U8 (decoded through
    the table at `lut_off`), Float32 linear RGB for FORMAT_F32. n_bytes == 0
    means the file didn't load."""
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var n_bytes: Int
    var width: Int32
    var height: Int32
    var n_levels: Int32
    var channels: Int32
    var format: Int32
    var lut_off: Int32

# Decode `filename` and build its full mip pyramid on the host: 8-bit files stay
# 8-bit (level 0 through load_texture_u8), everything else becomes Float32 RGB.
# `lut` holds both 256-entry decode tables (linear at 0, sRGB at 256) and `inv`
# their _build_inverse_lut tables (at 0 and _INV_LUT_SIZE). Reads the tables and
# touches only its own allocations, so it is safe to run on worker threads.
def _load_host_texture(
    filename: Pointer[UInt8, MutUntrackedOrigin], raw_flag: Int32,
    lut: Pointer[Float32, MutUntrackedOrigin], inv: Pointer[UInt8, MutUntrackedOrigin],
) -> _HostTexture:
    var result = _HostTexture(Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), 0,
                              Int32(0), Int32(0), Int32(0), Int32(0), Int32(GpuTexture.FORMAT_F32), Int32(0))
    var w_out = unsafe_alloc[Int32](1); var h_out = unsafe_alloc[Int32](1)
    var c_out = unsafe_alloc[Int32](1); var srgb_out = unsafe_alloc[Int32](1)
    w_out[unsafe_offset=0] = Int32(0); h_out[unsafe_offset=0] = Int32(0)
    var u8_out = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var ok_u8 = external_call["load_texture_u8", Int32,
        Pointer[UInt8, MutUntrackedOrigin], Int32,
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
        Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
        Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin]](
        filename, raw_flag, u8_out, w_out, h_out, c_out, srgb_out)
    if ok_u8 != 0 and Int(w_out[unsafe_offset=0]) > 0:
        var tw = Int(w_out[unsafe_offset=0]); var th = Int(h_out[unsafe_offset=0]); var c = Int(c_out[unsafe_offset=0])
        var lut_off = 256 if srgb_out[unsafe_offset=0] != Int32(0) else 0
        var (nlev, texels) = _mip_texel_count(tw, th)
        var pyr = unsafe_alloc[UInt8](texels * c)
        _fill_u8_mips(pyr, u8_out[unsafe_offset=0], tw, th, c, lut.unsafe_offset(lut_off),
                      inv.unsafe_offset((_INV_LUT_SIZE if lut_off != 0 else 0)))
        result = _HostTexture(pyr.unsafe_origin_cast[MutUntrackedOrigin](), texels * c, Int32(tw), Int32(th),
                              Int32(nlev), Int32(c), Int32(GpuTexture.FORMAT_U8), Int32(lut_off))
        _ = external_call["free_texture_u8", Int32, Pointer[UInt8, MutUntrackedOrigin]](u8_out[unsafe_offset=0])
    else:
        if ok_u8 != 0:
            _ = external_call["free_texture_u8", Int32, Pointer[UInt8, MutUntrackedOrigin]](u8_out[unsafe_offset=0])
        var data_out = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
        w_out[unsafe_offset=0] = Int32(0); h_out[unsafe_offset=0] = Int32(0)
        var ok = external_call["load_texture_rgb", Int32,
            Pointer[UInt8, MutUntrackedOrigin],
            Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
            Pointer[Int32, MutUntrackedOrigin],
            Pointer[Int32, MutUntrackedOrigin],
            Int32](filename, data_out, w_out, h_out, raw_flag)
        if ok != 0 and Int(w_out[unsafe_offset=0]) > 0:
            var tw = Int(w_out[unsafe_offset=0]); var th = Int(h_out[unsafe_offset=0])
            var (nlev, texels) = _mip_texel_count(tw, th)
            var pyr = unsafe_alloc[Float32](texels * 3)
            _fill_f32_mips(pyr, data_out[unsafe_offset=0], tw, th)
            result = _HostTexture(pyr.unsafe_bitcast[UInt8]().unsafe_origin_cast[MutUntrackedOrigin](), texels * 3 * 4,
                                  Int32(tw), Int32(th), Int32(nlev), Int32(3), Int32(GpuTexture.FORMAT_F32), Int32(0))
            _ = external_call["free_texture_rgb", Int32, Pointer[Float32, MutUntrackedOrigin]](data_out[unsafe_offset=0])
        data_out.unsafe_free()
    w_out.unsafe_free(); h_out.unsafe_free(); c_out.unsafe_free(); srgb_out.unsafe_free(); u8_out.unsafe_free()
    if result.n_bytes == 0:
        # A missing file is already reported at parse time (parse_types.mojo,
        # scene_path); this is the one that exists and still will not decode.
        # It used to be counted in "N unique file(s) loaded" and then render
        # as the material's or light's flat default without a word.
        print("Warning: could not decode texture '" + String(unsafe_from_utf8_ptr=filename.as_imm())
              + "' -- it renders as a flat default instead.")
    return result

@fieldwise_init
struct HostTextures(TrivialRegisterPassable):
    """Every texture of a scene decoded on the host. `tex[i]` is valid where
    `dup_of[i] == -1`; otherwise texture i shares the pixels of `dup_of[i]`.
    `lut` holds the two 256-entry uint8 decode tables (linear at 0, sRGB at 256)."""
    var tex: Pointer[_HostTexture, MutUntrackedOrigin]
    var dup_of: Pointer[Int32, MutUntrackedOrigin]
    var lut: Pointer[Float32, MutUntrackedOrigin]
    var count: Int

    def free_pixels(self):
        for ti in range(self.count):
            if self.dup_of[unsafe_offset=ti] == Int32(-1) and self.tex[unsafe_offset=ti].n_bytes > 0:
                self.tex[unsafe_offset=ti].data.unsafe_free()

    def free_tables(self):
        self.tex.unsafe_free(); self.dup_of.unsafe_free(); self.lut.unsafe_free()

def _tex_thread_main(arg: ThreadArg) -> ThreadArg:
    arg.unsafe_bitcast[TexPrefetch]()[unsafe_offset=0].run()
    return arg

struct TexPrefetch(Movable):
    """The scene's image textures, decoded on a few OS threads while the rest of the scene is still being built (the
    BVH build runs in parallel with it). Entries are keyed by (file, raw-ness); decode_host_textures takes the ones
    that match what it is asked for and decodes the rest itself, so a wrong guess of raw-ness costs only time.
    Plain pthreads, not the Mojo worker pool: a long-running pool task would stall every parallelize beside it."""
    var n: Int
    var files: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var raw: Pointer[Bool, MutUntrackedOrigin]
    var tex: Pointer[_HostTexture, MutUntrackedOrigin]
    var taken: Pointer[Bool, MutUntrackedOrigin]
    var cursor: Pointer[Int32, MutUntrackedOrigin]
    var lut: Pointer[Float32, MutUntrackedOrigin]
    var inv: Pointer[UInt8, MutUntrackedOrigin]
    var threads: List[OsThread]

    def __init__(out self):
        self.n = 0
        self.files = Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling()
        self.raw = Pointer[Bool, MutUntrackedOrigin].unsafe_dangling()
        self.tex = Pointer[_HostTexture, MutUntrackedOrigin].unsafe_dangling()
        self.taken = Pointer[Bool, MutUntrackedOrigin].unsafe_dangling()
        self.cursor = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        self.lut = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        self.inv = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
        self.threads = List[OsThread]()

    def run(self):
        while True:
            var k = Int(Atomic.fetch_add(self.cursor, Int32(1)))
            if k >= self.n:
                return
            self.tex[unsafe_offset=k] = _load_host_texture(self.files[unsafe_offset=k], Int32(1) if self.raw[unsafe_offset=k] else Int32(0),
                                                           self.lut, self.inv)

    def start(mut self, names: List[String], raws: List[Bool], max_threads: Int):
        """Decode the (name, raw) pairs on up to `max_threads` new threads and return at once."""
        var n = len(names)
        if n == 0:
            return
        self.n = n
        self.files = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](n)
        self.raw = unsafe_alloc[Bool](n)
        self.tex = unsafe_alloc[_HostTexture](n)
        self.taken = unsafe_alloc[Bool](n)
        for i in range(n):
            var nb = names[i].byte_length()
            var cp = unsafe_alloc[UInt8](nb + 1)
            for k in range(nb):
                cp[unsafe_offset=k] = names[i].as_bytes()[k]
            cp[unsafe_offset=nb] = UInt8(0)
            self.files[unsafe_offset=i] = cp
            self.raw[unsafe_offset=i] = raws[i]
            self.taken[unsafe_offset=i] = False
            self.tex[unsafe_offset=i] = _HostTexture(Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), 0,
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(GpuTexture.FORMAT_F32), Int32(0))
        self.cursor = unsafe_alloc[Int32](1)
        self.cursor[unsafe_offset=0] = Int32(0)
        self.lut = unsafe_alloc[Float32](512)
        _ = external_call["texture_uint8_lut", NoneType, Int32, Pointer[Float32, MutUntrackedOrigin]](Int32(0), self.lut)
        _ = external_call["texture_uint8_lut", NoneType, Int32, Pointer[Float32, MutUntrackedOrigin]](Int32(1), self.lut.unsafe_offset(256))
        self.inv = unsafe_alloc[UInt8](2 * _INV_LUT_SIZE)
        _build_inverse_lut(self.lut, self.inv)
        _build_inverse_lut(self.lut.unsafe_offset(256), self.inv.unsafe_offset(_INV_LUT_SIZE))
        var self_ptr = Pointer(to=self).unsafe_bitcast[UInt8]().unsafe_origin_cast[MutUntrackedOrigin]()
        for t in range(min(max_threads, n)):
            self.threads.append(OsThread())
            self.threads[t].start(_tex_thread_main, self_ptr)

    def join(mut self):
        for t in range(len(self.threads)):
            self.threads[t].join()
        self.threads.clear()

    def find(self, filename: Pointer[UInt8, MutUntrackedOrigin], raw: Bool) -> Int:
        for i in range(self.n):
            if self.raw[unsafe_offset=i] == raw and _cstr_eq(self.files[unsafe_offset=i], filename):
                return i
        return -1

    def free_rest(mut self):
        """Release everything the decode did not take (called once decode_host_textures has had its pick)."""
        if self.n == 0:
            return
        for i in range(self.n):
            if not self.taken[unsafe_offset=i] and self.tex[unsafe_offset=i].n_bytes > 0:
                self.tex[unsafe_offset=i].data.unsafe_free()
            self.files[unsafe_offset=i].unsafe_free()
        self.files.unsafe_free(); self.raw.unsafe_free(); self.tex.unsafe_free(); self.taken.unsafe_free()
        self.cursor.unsafe_free(); self.lut.unsafe_free(); self.inv.unsafe_free()
        self.n = 0

def decode_host_textures(
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin], n_textures: Int,
    materials: Pointer[Material, MutUntrackedOrigin], n_materials: Int,
    prefetch: Pointer[TexPrefetch, MutUntrackedOrigin] = Pointer[TexPrefetch, MutUntrackedOrigin].unsafe_dangling(),
) -> HostTextures:
    # Textures referenced as normal maps hold linear data and must NOT be
    # sRGB-decoded on load. Mark those indices by scanning the materials.
    var tex_is_raw = unsafe_alloc[Bool](max(n_textures, 1))
    for ti in range(n_textures):
        tex_is_raw[unsafe_offset=ti] = False
    for mi in range(n_materials):
        var nidx = Int(materials[unsafe_offset=mi].normal_tex_idx)
        if nidx >= 0 and nidx < n_textures:
            tex_is_raw[unsafe_offset=nidx] = True
    # Many scenes (e.g. landscape) declare a separate named Texture per
    # instance even when several instances share the same underlying
    # image file (batch-exported "-renamed-N" duplicates). Dedup by
    # (filename, raw-ness) so each unique file is only loaded from disk
    # once, instead of once per declaration.
    var dup_of = unsafe_alloc[Int32](max(n_textures, 1))
    for ti in range(n_textures):
        dup_of[unsafe_offset=ti] = Int32(-1)
        for tj in range(ti):
            if dup_of[unsafe_offset=tj] == Int32(-1) and tex_is_raw[unsafe_offset=tj] == tex_is_raw[unsafe_offset=ti] and \
               _cstr_eq(tex_filenames[unsafe_offset=ti], tex_filenames[unsafe_offset=tj]):
                dup_of[unsafe_offset=ti] = Int32(tj)
                break
    # 8-bit textures stay 8-bit and decode through one of two 256-entry
    # tables (linear at 0, sRGB at 256), built by the oiio bridge exactly as
    # load_texture_rgb decodes, so level 0 matches the float path.
    var lut_host = unsafe_alloc[Float32](512)
    _ = external_call["texture_uint8_lut", NoneType, Int32, Pointer[Float32, MutUntrackedOrigin]](Int32(0), lut_host)
    _ = external_call["texture_uint8_lut", NoneType, Int32, Pointer[Float32, MutUntrackedOrigin]](Int32(1), lut_host.unsafe_offset(256))
    var inv_host = unsafe_alloc[UInt8](2 * _INV_LUT_SIZE)
    _build_inverse_lut(lut_host, inv_host)
    _build_inverse_lut(lut_host.unsafe_offset(256), inv_host.unsafe_offset(_INV_LUT_SIZE))
    # Decoding and mip building are independent per file and dominate
    # startup on texture-heavy scenes (Bistro), so run them on every core,
    # workers claiming the next texture from a shared cursor (file sizes
    # vary a lot).
    var host_tex = unsafe_alloc[_HostTexture](max(n_textures, 1))
    var next_tex = unsafe_alloc[Int32](1)
    next_tex[unsafe_offset=0] = Int32(0)

    var have_prefetch = Int(prefetch) > 8 and prefetch[unsafe_offset=0].n > 0

    def decode_worker(_worker_idx: Int) {imm}:
        while True:
            var ti = Int(Atomic.fetch_add(next_tex, Int32(1)))
            if ti >= n_textures:
                break
            if dup_of[unsafe_offset=ti] == Int32(-1):
                var raw_flag = Int32(1) if tex_is_raw[unsafe_offset=ti] else Int32(0)
                var hit = -1
                if have_prefetch:
                    hit = prefetch[unsafe_offset=0].find(tex_filenames[unsafe_offset=ti], tex_is_raw[unsafe_offset=ti])
                if hit >= 0:
                    host_tex[unsafe_offset=ti] = prefetch[unsafe_offset=0].tex[unsafe_offset=hit]
                    prefetch[unsafe_offset=0].taken[unsafe_offset=hit] = True
                else:
                    host_tex[unsafe_offset=ti] = _load_host_texture(tex_filenames[unsafe_offset=ti], raw_flag, lut_host, inv_host)

    if n_textures > 0:
        parallelize(decode_worker, min(num_performance_cores(), n_textures))
    next_tex.unsafe_free()
    inv_host.unsafe_free()
    tex_is_raw.unsafe_free()
    if have_prefetch:
        prefetch[unsafe_offset=0].free_rest()
    return HostTextures(host_tex.unsafe_origin_cast[MutUntrackedOrigin](), dup_of.unsafe_origin_cast[MutUntrackedOrigin](),
                        lut_host.unsafe_origin_cast[MutUntrackedOrigin](), n_textures)

# The table shading.mojo samples, for the CPU renderer: entries point straight
# at the host pyramids, which must outlive it.
def host_texture_table(ht: HostTextures) -> Pointer[GpuTexture, MutUntrackedOrigin]:
    var table = unsafe_alloc[GpuTexture](max(ht.count, 1))
    for ti in range(ht.count):
        if ht.dup_of[unsafe_offset=ti] != Int32(-1):
            table[unsafe_offset=ti] = table[unsafe_offset=Int(ht.dup_of[unsafe_offset=ti])]
            continue
        var t = ht.tex[unsafe_offset=ti]
        if t.n_bytes == 0:
            table[unsafe_offset=ti] = GpuTexture(Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
                Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(GpuTexture.FORMAT_F32))
            continue
        table[unsafe_offset=ti] = GpuTexture(t.data, ht.lut.unsafe_offset(Int(t.lut_off)),
            t.width, t.height, t.n_levels, t.channels, t.format)
    return table.unsafe_origin_cast[MutUntrackedOrigin]()
