from std.ffi import external_call

@export
fn mojo_init():
    var msg = "Gonzales Mojo Kernel Initialized!\n"
    _ = external_call["printf", Int32](msg.unsafe_ptr())
