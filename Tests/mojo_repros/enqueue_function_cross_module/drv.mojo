# Fails: this module imports nothing from max.gpu.host.
# Compiles once any name is imported, e.g. `from max.gpu.host import DeviceBuffer`.
from hmod import H


def kern():
    pass


def main() raises:
    var h = H()
    h.ctx.enqueue_function[kern](grid_dim=1, block_dim=1)
