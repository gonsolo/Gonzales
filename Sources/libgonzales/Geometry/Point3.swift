public struct Point3: Sendable, ThreeComponent {

        init() {
                self.init(x: 0, y: 0, z: 0)
        }

        public init(x: Real, y: Real, z: Real) {
                self.xyz = SIMD4<Real>(x, y, z, 1.0)
        }

        init(_ point: Point3) {
                self.xyz = SIMD4<Real>(point.x, point.y, point.z, 1.0)
        }

        init(xyz: (Real, Real, Real)) {
                self.xyz = SIMD4<Real>(xyz.0, xyz.1, xyz.2, 1.0)
        }

        init(_ normal: Normal3) {
                self.init(
                        x: normal.x,
                        y: normal.y,
                        z: normal.z)
        }

        subscript(index: Int) -> Real {
                get { return xyz[index] }
                set(newValue) { xyz[index] = newValue }
        }

        public var x: Real {
                get { return xyz.x }
                set { xyz.x = newValue }
        }

        public var y: Real {
                get { return xyz.y }
                set { xyz.y = newValue }
        }

        public var z: Real {
                get { return xyz.z }
                set { xyz.z = newValue }
        }

        public static func * (mul: Real, point: Point3) -> Point3 {
                var result = Point3()
                result.xyz = point.xyz * mul
                return result
        }

        public static func / (point: Point3, divisor: Real) -> Point3 {
                var result = Point3()
                result.xyz = point.xyz / divisor
                return result
        }

        public static func + (left: Point3, right: Point3) -> Point3 {
                var result = Point3()
                result.xyz = left.xyz + right.xyz
                return result
        }

        public static func + (left: Point3, right: Vector3) -> Point3 {
                var result = Point3()
                result.xyz = left.xyz + right.xyz
                return result
        }

        public static func - (left: Point3, right: Point3) -> Point3 {
                var result = Point3()
                result.xyz = left.xyz - right.xyz
                return result
        }

        public static func - (left: Point3, right: Point3) -> Vector3 {
                var result = Vector3()
                result.xyz = left.xyz - right.xyz
                return result
        }

        var xyz: SIMD4<Real>
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
