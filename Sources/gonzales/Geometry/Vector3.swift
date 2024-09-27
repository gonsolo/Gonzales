public struct Vector3<T: FloatingPoint & Sendable>: Sendable, Three {

        public init(x: T, y: T, z: T) {
                self.x = x
                self.y = y
                self.z = z
        }

        init(v: T) {
                self.x = v
                self.y = v
                self.z = v
        }

        init() {
                self.x = 0
                self.y = 0
                self.z = 1
        }

        init(vector: Vector3) {
                self.x = vector.x
                self.y = vector.y
                self.z = vector.z
        }

        init(xyz: (T, T, T)) {
                self.x = xyz.0
                self.y = xyz.1
                self.z = xyz.2
        }

        subscript(index: Int) -> T {
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

        public var x: T
        public var y: T
        public var z: T
}

extension Vector3 {
        init(point: Point3<T>) {
                self.x = point.x
                self.y = point.y
                self.z = point.z
        }
        init(normal: Normal3<T>) {
                self.x = normal.x
                self.y = normal.y
                self.z = normal.z
        }
}

extension Vector3: CustomStringConvertible {
        public var description: String {
                return "(\(x) \(y) \(z))"
        }
}

func cross(_ a: Vector, _ b: Vector) -> Vector {
        return Vector(
                x: a.y * b.z - a.z * b.y,
                y: a.z * b.x - a.x * b.z,
                z: a.x * b.y - a.y * b.x)
}

public typealias Vector = Vector3<FloatX>

let nullVector = Vector(x: 0, y: 0, z: 0)
let up = Vector(x: 0, y: 1, z: 0)

func mirror<T: FloatingPoint>(_ vector: Vector3<T>) -> Vector3<T> {
        return Vector3(x: -vector.x, y: -vector.y, z: vector.z)
}

func abs<T: FloatingPoint>(_ vector: Vector3<T>) -> Vector3<T> {
        return Vector3(x: abs(vector.x), y: abs(vector.y), z: abs(vector.z))
}

func permute<T>(vector: Vector3<T>, x: Int, y: Int, z: Int) -> Vector3<T> {
        return Vector3(x: vector[x], y: vector[y], z: vector[z])
}
