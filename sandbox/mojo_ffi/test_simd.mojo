fn main():
    var a = SIMD[DType.float32, 4](1, 2, 3, 4)
    var b = SIMD[DType.float32, 4](4, 3, 2, 1)
    # var c = a <= b
    var d = a <= b
    var e = a.le(b)
