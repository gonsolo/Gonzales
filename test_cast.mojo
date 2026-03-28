fn main():\n    var ptr = UnsafePointer[UInt8].alloc(10)\n    var p2 = ptr.bitcast[Float32]()\n    print(1)
