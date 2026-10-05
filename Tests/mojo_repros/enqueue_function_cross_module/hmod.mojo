from max.gpu.host import DeviceContext


struct H(Movable):
    var ctx: DeviceContext

    def __init__(out self) raises:
        self.ctx = DeviceContext()
