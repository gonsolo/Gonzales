from std.ffi import external_call
from std.memory.alloc import unsafe_alloc

# A plain OS thread (pthread), for work that must neither run in the Mojo worker pool (a long task there stalls every
# parallelize beside it) nor under the pool's flush-denormals mode (it would change float results).
comptime ThreadArg = Pointer[UInt8, MutUntrackedOrigin]

struct OsThread(Movable):
    var tid: Pointer[UInt64, MutUntrackedOrigin]
    var started: Bool

    def __init__(out self):
        self.tid = unsafe_alloc[UInt64](1)
        self.started = False

    def start(mut self, entry: def(ThreadArg) thin -> ThreadArg, arg: ThreadArg):
        """Run entry(arg) on a new thread; returns at once. If no thread can be created, entry runs here."""
        var attr = unsafe_alloc[UInt8](128)
        _ = external_call["pthread_attr_init", Int32](attr)
        self.started = external_call["pthread_create", Int32](self.tid, attr, entry, arg) == Int32(0)
        attr.unsafe_free()
        if not self.started:
            _ = entry(arg)

    def join(mut self):
        if self.started:
            var rv = unsafe_alloc[ThreadArg](1)
            _ = external_call["pthread_join", Int32](self.tid[unsafe_offset=0], rv)
            rv.unsafe_free()
            self.started = False
