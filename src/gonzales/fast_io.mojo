"""Reading a big array from a file on every core: the cache loads are memory-bound, and one fread runs at a few GB/s."""
from max.algorithm import parallelize
from std.atomic import Atomic
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores

comptime _Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime READ_CHUNK = 16 * 1024 * 1024
comptime PARALLEL_MIN = 32 * 1024 * 1024


def read_parallel(fd: Int32, dst: _Ptr, nbytes: Int, offset: Int):
    """pread nbytes at file offset into dst, 16 MB pieces claimed by up to 12 workers."""
    var n_chunks = (nbytes + READ_CHUNK - 1) // READ_CHUNK
    var cursor = unsafe_alloc[Int32](1)
    cursor[unsafe_offset=0] = Int32(0)

    def worker(_w: Int) {imm}:
        while True:
            var c = Int(Atomic.fetch_add(cursor, Int32(1)))
            if c >= n_chunks:
                break
            var o = c * READ_CHUNK
            var length = min(READ_CHUNK, nbytes - o)
            var done = 0
            while done < length:
                var r = external_call["pread", Int, Int32, _Ptr, Int, Int](fd, dst.unsafe_offset(o + done), length - done, offset + o + done)
                if r <= 0:
                    break
                done += r

    parallelize(worker, max(min(min(num_performance_cores(), 12), n_chunks), 1))
    cursor.unsafe_free()
