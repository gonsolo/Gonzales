# Device buffers whose requested bytes libxert counts, for the memory table of GONZALES_GPU_LOG.
from std.ffi import external_call
from max.gpu.host import DeviceBuffer, DeviceContext


def dev_buf(ctx: DeviceContext, nbytes: Int) raises -> DeviceBuffer[DType.uint8]:
    external_call["gz_acct_add", NoneType](UInt64(nbytes))
    return ctx.enqueue_create_buffer[DType.uint8](nbytes)


def dev_requested() -> Int:
    return Int(external_call["gz_acct_bytes", UInt64]())
