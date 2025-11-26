public struct Normal3: Sendable, Three {

        init() {
                self.x = 0
                self.y = 1
                self.z = 0
        }

        public init(x: FloatX = 0, y: FloatX = 0, z: FloatX = 1) {
                self.x = x
                self.y = y
                self.z = z
        }

        init(_ normal: Normal3) {
                self.x = normal.x
                self.y = normal.y
                self.z = normal.z
        }

        init(point: Point3) {
                self.x = point.x
                self.y = point.y
                self.z = point.z
        }

        init(_ vector: Vector3) {
                self.x = vector.x
                self.y = vector.y
                self.z = vector.z
        }

        init(xyz: (FloatType, FloatType, FloatType)) {
                self.x = xyz.0
                self.y = xyz.1
                self.z = xyz.2
        }

        var x: FloatX
        var y: FloatX
        var z: FloatX
}

extension Normal3: CustomStringConvertible {
        public var description: String {
                return "[ \(x) \(y) \(z)]"
        }
}

public typealias Normal = Normal3

let upNormal = Normal(x: 0, y: 0, z: 1)
let zeroNormal = Normal(x: 0, y: 0, z: 0)
