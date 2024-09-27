public struct Normal3<T: FloatingPoint & Sendable>: Sendable, Three {

        init() {
                self.x = 0
                self.y = 1
                self.z = 0
        }

        init(x: T = 0, y: T = 0, z: T = 1) {
                self.x = x
                self.y = y
                self.z = z
        }

        init(_ normal: Normal3<T>) {
                self.x = normal.x
                self.y = normal.y
                self.z = normal.z
        }

        init(point: Point3<T>) {
                self.x = point.x
                self.y = point.y
                self.z = point.z
        }

        init(_ vector: Vector3<T>) {
                self.x = vector.x
                self.y = vector.y
                self.z = vector.z
        }

        init(xyz: (T, T, T)) {
                self.x = xyz.0
                self.y = xyz.1
                self.z = xyz.2
        }

        var x: T
        var y: T
        var z: T
}

extension Normal3: CustomStringConvertible {
        public var description: String {
                return "[ \(x) \(y) \(z)]"
        }
}

public typealias Normal = Normal3<FloatX>

let upNormal = Normal(x: 0, y: 0, z: 1)
let zeroNormal = Normal(x: 0, y: 0, z: 0)
