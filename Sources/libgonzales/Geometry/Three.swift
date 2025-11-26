/// A type that provides the base for all types consisting of three
/// values like points and normals.

public protocol Three {

        associatedtype FloatType: FloatingPoint

        init(x: FloatType, y: FloatType, z: FloatType)

        var x: FloatType { get set }
        var y: FloatType { get set }
        var z: FloatType { get set }
}

func lengthSquared<T: Three>(_ v: T) -> T.FloatType {
        return v.x * v.x + v.y * v.y + v.z * v.z
}

func length<T: Three>(_ v: T) -> T.FloatType {
        return lengthSquared(v).squareRoot()
}

func distanceSquared<T: Three>(_ left: T, _ right: T) -> T.FloatType {
        return lengthSquared(left - right)
}

func distance<T: Three>(_ left: T, _ right: T) -> T.FloatType {
        return distanceSquared(left, right).squareRoot()
}

prefix func - <T: Three>(v: T) -> T {
        return T.init(x: -v.x, y: -v.y, z: -v.z)
}

func / <T: Three>(v: T, d: T.FloatType) -> T {
        return T.init(x: v.x / d, y: v.y / d, z: v.z / d)
}

func / <T: Three>(v: T, d: T) -> T {
        return T.init(x: v.x / d.x, y: v.y / d.y, z: v.z / d.z)
}

func * <T: Three>(left: T, right: T) -> T {
        return T.init(
                x: left.x * right.x,
                y: left.y * right.y,
                z: left.z * right.z)
}

func + <T: Three>(left: T, right: T) -> T {
        return T.init(
                x: left.x + right.x,
                y: left.y + right.y,
                z: left.z + right.z)
}

func - <T: Three>(left: T, right: T) -> T {
        return T.init(
                x: left.x - right.x,
                y: left.y - right.y,
                z: left.z - right.z)
}

func * <T: Three>(left: T, right: T.FloatType) -> T {
        return T.init(
                x: left.x * right,
                y: left.y * right,
                z: left.z * right)
}

func * <T: Three>(left: T.FloatType, right: T) -> T {
        return right * left
}

func += <T: Three>(left: inout T, right: T) {
        left.x += right.x
        left.y += right.y
        left.z += right.z
}

func *= <T: Three>(left: inout T, right: T.FloatType) {
        left.x *= right
        left.y *= right
        left.z *= right
}

func /= <T: Three>(left: inout T, right: T.FloatType) {
        left.x /= right
        left.y /= right
        left.z /= right
}

func == <T: Three>(left: T, right: T) -> Bool {
        return left.x == right.x && left.y == right.y && left.z == right.z
}

func != <T: Three>(left: T, right: T) -> Bool {
        return !(left == right)
}

extension Three {
        public mutating func normalize() {
                self = normalized(self)
        }
}

public func normalized<T: Three>(_ v: T) -> T {
        var ret: T
        let l = length(v)
        if l == 0 {
                ret = v
        } else {
                ret = v / l
        }
        return ret
}

public func cross<T: Three>(_ a: T, _ b: T) -> T
where T.FloatType: FloatingPoint {

        // Ensure all arithmetic operations (multiplication, subtraction) use the correct FloatType
        let x = a.y * b.z - a.z * b.y
        let y = a.z * b.x - a.x * b.z
        let z = a.x * b.y - a.y * b.x

        // Create a new instance of the generic type T
        return T(x: x, y: y, z: z)
}

func dot<T: Three>(_ a: T, _ b: T) -> T.FloatType {
        return a.x * b.x + a.y * b.y + a.z * b.z
}

func absDot<T: Three>(_ a: T, _ b: T) -> T.FloatType {
        return abs(dot(a, b))
}
