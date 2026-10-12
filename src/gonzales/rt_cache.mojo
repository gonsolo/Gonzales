"""A binary snapshot of the finished RT-core structures (BLASes, TLAS, id tables), so a restart skips the builders.

GONZALES_RT_CACHE=<file> (default: the scene cache's name plus .rt). The key hashes everything the structures depend on
(triangle counts, instance matrices); a different key means the file is rebuilt.
"""
from std.memory.alloc import unsafe_alloc
from .scene_cache import _fopen, _fclose, _put, _get, _Ptr

comptime _RT_MAGIC: Int = 0x435452544E5A48
comptime PB = Pointer[UInt8, MutUntrackedOrigin]
comptime PBP = Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
comptime PIP = Pointer[Pointer[Int32, MutUntrackedOrigin], MutUntrackedOrigin]


def _c_path(path: String) -> _Ptr:
    var p = unsafe_alloc[UInt8](path.byte_length() + 1)
    for k in range(path.byte_length()):
        p[unsafe_offset=k] = path.unsafe_ptr()[unsafe_offset=k]
    p[unsafe_offset=path.byte_length()] = UInt8(0)
    return p


def rt_cache_save(
    path: String, key: Int, n_blas: Int, blas_bytes: PBP, blas_sizes: Pointer[Int64, MutUntrackedOrigin],
    blas_box: Pointer[Float64, MutUntrackedOrigin], blas_gid: PIP, blas_ngid: Pointer[Int, MutUntrackedOrigin],
    tlas_bytes: PB, tlas_size: Int, rec_off: Int, n_records: Int, inst_of_record: Pointer[Int32, MutUntrackedOrigin],
) -> Bool:
    var cp = _c_path(path)
    var f = _fopen(cp, True)
    cp.unsafe_free()
    if f == 0:
        return False
    var head = unsafe_alloc[Int](6)
    head[unsafe_offset=0] = _RT_MAGIC; head[unsafe_offset=1] = key; head[unsafe_offset=2] = n_blas
    head[unsafe_offset=3] = tlas_size; head[unsafe_offset=4] = rec_off; head[unsafe_offset=5] = n_records
    _put[Int](f, head, 6)
    head.unsafe_free()
    _put[Int64](f, blas_sizes, n_blas); _put[Int](f, blas_ngid, n_blas); _put[Float64](f, blas_box, 6 * n_blas)
    for b in range(n_blas):
        _put[UInt8](f, blas_bytes[unsafe_offset=b], Int(blas_sizes[unsafe_offset=b]))
        _put[Int32](f, blas_gid[unsafe_offset=b], blas_ngid[unsafe_offset=b])
    _put[UInt8](f, tlas_bytes, tlas_size)
    _put[Int32](f, inst_of_record, n_records)
    _fclose(f)
    return True


# Fills the caller's arrays (allocated for n_blas) and returns the TLAS; an empty result (size 0) means no usable cache.
struct RtCacheTlas(Movable):
    var bytes: PB
    var size: Int
    var rec_off: Int
    var n_records: Int
    var inst_of_record: Pointer[Int32, MutUntrackedOrigin]     # record -> instance index of the build order

    def __init__(out self, bytes: PB, size: Int, rec_off: Int, n_records: Int, inst_of_record: Pointer[Int32, MutUntrackedOrigin]):
        self.bytes = bytes
        self.size = size
        self.rec_off = rec_off
        self.n_records = n_records
        self.inst_of_record = inst_of_record


def rt_cache_load(
    path: String, key: Int, n_blas: Int, blas_bytes: PBP, blas_sizes: Pointer[Int64, MutUntrackedOrigin],
    blas_box: Pointer[Float64, MutUntrackedOrigin], blas_gid: PIP, blas_ngid: Pointer[Int, MutUntrackedOrigin],
) -> RtCacheTlas:
    var none = RtCacheTlas(unsafe_alloc[UInt8](1), 0, 0, 0, unsafe_alloc[Int32](1))
    var cp = _c_path(path)
    var f = _fopen(cp, False)
    cp.unsafe_free()
    if f == 0:
        return none^
    var head = _get[Int](f, 6)
    var ok = head[unsafe_offset=0] == _RT_MAGIC and head[unsafe_offset=1] == key and head[unsafe_offset=2] == n_blas
    var tlas_size = head[unsafe_offset=3]; var rec_off = head[unsafe_offset=4]; var n_records = head[unsafe_offset=5]
    head.unsafe_free()
    if not ok:
        _fclose(f)
        return none^
    var sz = _get[Int64](f, n_blas); var ng = _get[Int](f, n_blas); var bx = _get[Float64](f, 6 * n_blas)
    for b in range(n_blas):
        blas_sizes[unsafe_offset=b] = sz[unsafe_offset=b]; blas_ngid[unsafe_offset=b] = ng[unsafe_offset=b]
    for q in range(6 * n_blas):
        blas_box[unsafe_offset=q] = bx[unsafe_offset=q]
    sz.unsafe_free(); ng.unsafe_free(); bx.unsafe_free()
    for b in range(n_blas):
        blas_bytes[unsafe_offset=b] = _get[UInt8](f, Int(blas_sizes[unsafe_offset=b]))
        blas_gid[unsafe_offset=b] = _get[Int32](f, blas_ngid[unsafe_offset=b])
    var tl = _get[UInt8](f, tlas_size)
    var perm = _get[Int32](f, n_records)
    _fclose(f)
    none.bytes.unsafe_free(); none.inst_of_record.unsafe_free()
    return RtCacheTlas(tl, tlas_size, rec_off, n_records, perm)
