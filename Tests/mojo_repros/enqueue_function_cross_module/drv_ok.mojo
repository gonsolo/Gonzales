# Same as drv.mojo plus one unused import from max.gpu.host: compiles.
from max.gpu.host import DeviceBuffer
from hmod import H


def kern():
    pass


def main() raises:
    var h = H()
    h.ctx.enqueue_function[kern](grid_dim=1, block_dim=1)
