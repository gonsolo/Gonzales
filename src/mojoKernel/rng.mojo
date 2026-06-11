# ── Random Number Generation ────────────────────────────────────────

struct PCG32:
    var state: UInt64
    var inc: UInt64

    def __init__(out self, initstate: UInt64, initseq: UInt64):
        self.state = 0
        self.inc = (initseq << 1) | 1
        _ = self.next_uint()
        self.state += initstate
        _ = self.next_uint()

    def next_uint(mut self) -> UInt32:
        var oldstate = self.state
        self.state = oldstate * 6364136223846793005 + self.inc
        var xorshifted = UInt32(((oldstate >> 18) ^ oldstate) >> 27)
        var rot = UInt32(oldstate >> 59)
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))

    def next_float(mut self) -> Float32:
        return Float32(self.next_uint() >> 8) * (1.0 / 16777216.0)
