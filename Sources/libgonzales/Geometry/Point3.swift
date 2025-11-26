public struct Point3: Sendable, Three {

        init() {
                self.init(x: 0, y: 0, z: 0)
        }

        public init(x: FloatX, y: FloatX, z: FloatX) {
                self.xyz = SIMD4<FloatX>(x, y, z, 1.0)
        }

        init(_ point: Point3) {
                self.xyz = SIMD4<FloatX>(point.x, point.y, point.z, 1.0)
        }

        init(xyz: (FloatX, FloatX, FloatX)) {
                self.xyz = SIMD4<FloatX>(xyz.0, xyz.1, xyz.2, 1.0)
        }

        init(_ normal: Normal3) {
                self.init(
                        x: normal.x,
                        y: normal.y,
                        z: normal.z)
        }

        subscript(index: Int) -> FloatX {
                get {
                        switch index {
                        case 0: return xyz.x
                        case 1: return xyz.y
                        case 2: return xyz.z
                        default: return xyz.x
                        }
                }

                set(newValue) {
                        switch index {
                        case 0: xyz.x = newValue
                        case 1: xyz.y = newValue
                        case 2: xyz.z = newValue
                        default: xyz.x = newValue
                        }
                }
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

        public static func * (mul: FloatX, point: Point3) -> Point3 {
                return Point3(x: point.x * mul, y: point.y * mul, z: point.z * mul)
        }

        public static func / (point: Point3, divisor: FloatX) -> Point3 {
                return Point3(x: point.x / divisor, y: point.y / divisor, z: point.z / divisor)
        }

        public static func + (left: Point3, right: Point3) -> Point3 {
                return Point3(x: left.x + right.x, y: left.y + right.y, z: left.z + right.z)
        }

        public static func + (left: Point3, right: Vector3) -> Point3 {
                return Point3(x: left.x + right.x, y: left.y + right.y, z: left.z + right.z)
        }

        public static func - (left: Point3, right: Point3) -> Point3 {
                return Point3(x: left.x - right.x, y: left.y - right.y, z: left.z - right.z)
        }

        public static func - (left: Point3, right: Point3) -> Vector3 {
                return Vector3(x: left.x - right.x, y: left.y - right.y, z: left.z - right.z)
        }

        var xyz: SIMD4<FloatX>
}

extension Point3: CustomStringConvertible {
        public var description: String {
                return "[ \(x) \(y) \(z) ]"
        }
}

func permute(point: Point3, x: Int, y: Int, z: Int) -> Point3 {
        return Point3(x: point[x], y: point[y], z: point[z])
}

public typealias Point = Point3

let origin = Point()
