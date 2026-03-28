fn main():
    var a = 10
    print("Hello " + String(a))
    
    var u = UInt(100)
    var i = Int(u)
    
    var ptr = UnsafePointer[UInt8]()
    var p2 = ptr.bitcast[Float32]()
    print(1)
