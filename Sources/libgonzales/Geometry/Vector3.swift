public struct Vector3: Sendable, Three {

        public init(x: FloatX, y: FloatX, z: FloatX) {
                self.xyz = SIMD4<FloatX>(x, y, z, 1.0)
        }

        init(v: FloatX) {
                self.xyz = SIMD4<FloatX>(v, v, v, 1.0)
        }

        init() {
                self.init(x: 0, y: 0, z: 1)
        }

        init(vector: Vector3) {
                self.init(x: vector.x, y: vector.y, z: vector.z)
        }

        init(xyz: (FloatX, FloatX, FloatX)) {
                self.init(x: xyz.0, y: xyz.1, z: xyz.2)
        }

        subscript(index: Int) -> FloatX {
                get {
                        switch index {
                        case 0: return x
                        case 1: return y
                        case 2: return z
                        default: return x
                        }
                }
                set(newValue) {
                        switch index {
                        case 0: x = newValue
                        case 1: y = newValue
                        case 2: z = newValue
                        default: x = newValue
                        }
                }
        }

        var isNaN: Bool {
                return x.isNaN || y.isNaN || z.isNaN
        }

        var isZero: Bool {
                return x.isZero && y.isZero && z.isZero
        }

        public var x: FloatX {
                get { return xyz.x }
                set { xyz.x = newValue }
        }

        public var y: FloatX {
                get { return xyz.y }
                set { xyz.y = newValue }
        }

        public var z: FloatX {
                get { return xyz.z }
                set { xyz.z = newValue }
        }

        var xyz: SIMD4<FloatX>
}

extension Vector3 {
        init(point: Point3) {
                self.init(x: point.x, y: point.y, z: point.z)
        }
}

extension Vector3 {
        init(normal: Normal3) {
                self.init(x: normal.x, y: normal.y, z: normal.z)
        }
}

extension Vector3: CustomStringConvertible {
        public var description: String {
                return "(\(x) \(y) \(z))"
        }
}

public typealias Vector = Vector3

let nullVector = Vector(x: 0, y: 0, z: 0)
let upVector = Vector(x: 0, y: 1, z: 0)

func mirror(_ vector: Vector3) -> Vector3 {
        return Vector3(x: -vector.x, y: -vector.y, z: vector.z)
}

func abs(_ vector: Vector3) -> Vector3 {
        return Vector3(x: abs(vector.x), y: abs(vector.y), z: abs(vector.z))
}

func permute(vector: Vector3, x: Int, y: Int, z: Int) -> Vector3 {
        return Vector3(x: vector[x], y: vector[y], z: vector[z])
}
