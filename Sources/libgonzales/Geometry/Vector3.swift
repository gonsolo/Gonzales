// @doc:vector3-struct
public struct Vector3: Sendable, ThreeComponent {

        public init(x: Real, y: Real, z: Real) {
                self.xyz = SIMD4<Real>(x, y, z, 1.0)
        }

        init(value: Real) {
                self.xyz = SIMD4<Real>(value, value, value, 1.0)
        }

        init() {
                self.init(x: 0, y: 0, z: 1)
        }

        init(vector: Vector3) {
                self.init(x: vector.x, y: vector.y, z: vector.z)
        }

        init(xyz: (Real, Real, Real)) {
                self.init(x: xyz.0, y: xyz.1, z: xyz.2)
        }

        subscript(index: Int) -> Real {
                get { return xyz[index] }
                set(newValue) { xyz[index] = newValue }
        }

        var isNaN: Bool {
                return x.isNaN || y.isNaN || z.isNaN
        }

        var isZero: Bool {
                return x.isZero && y.isZero && z.isZero
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

        var xyz: SIMD4<Real>
}
// @doc:end

extension Vector3 {
        public init(point: Point3) {
                self.init(x: point.x, y: point.y, z: point.z)
        }

        public static func + (left: Vector3, right: Vector3) -> Vector3 {
                var result = Vector3()
                result.xyz = left.xyz + right.xyz
                return result
        }

        public static func - (left: Vector3, right: Vector3) -> Vector3 {
                var result = Vector3()
                result.xyz = left.xyz - right.xyz
                return result
        }

        public static func * (left: Vector3, right: Vector3) -> Vector3 {
                var result = Vector3()
                result.xyz = left.xyz * right.xyz
                return result
        }

        public static func * (left: Vector3, right: Real) -> Vector3 {
                var result = Vector3()
                result.xyz = left.xyz * right
                return result
        }

        public static func * (left: Real, right: Vector3) -> Vector3 {
                var result = Vector3()
                result.xyz = left * right.xyz
                return result
        }

        public static func / (left: Vector3, right: Real) -> Vector3 {
                var result = Vector3()
                result.xyz = left.xyz / right
                return result
        }

        public static func / (left: Vector3, right: Vector3) -> Vector3 {
                var result = Vector3()
                result.xyz = left.xyz / right.xyz
                return result
        }

        public static prefix func - (vector: Vector3) -> Vector3 {
                var result = Vector3()
                result.xyz = -vector.xyz
                return result
        }

        public static func += (left: inout Vector3, right: Vector3) {
                left.xyz += right.xyz
        }

        public static func -= (left: inout Vector3, right: Vector3) {
                left.xyz -= right.xyz
        }

        public static func *= (left: inout Vector3, right: Real) {
                left.xyz *= right
        }

        public static func /= (left: inout Vector3, right: Real) {
                left.xyz /= right
        }
}

extension Vector3 {
        public init(normal: Normal3) {
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
