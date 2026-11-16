# Device buffers whose requested bytes libxert counts, for the memory table of GONZALES_GPU_LOG.
from std.ffi import external_call
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext


def dev_buf[dtype: DType = DType.uint8](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dtype]:
    external_call["gz_acct_add", NoneType](UInt64(n * size_of[Scalar[dtype]]()))
    return ctx.enqueue_create_buffer[dtype](n)


def dev_requested() -> Int:
    return Int(external_call["gz_acct_bytes", UInt64]())
