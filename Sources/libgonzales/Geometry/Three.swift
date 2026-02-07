/// A type that provides the base for all types consisting of three
/// values like points and normals.

public protocol Three {

        associatedtype FloatType: FloatingPoint

        init(x: FloatType, y: FloatType, z: FloatType)

        var x: FloatType { get set }
        var y: FloatType { get set }
        var z: FloatType { get set }
}

func lengthSquared<T: Three>(_ vector: T) -> T.FloatType {
        return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z
}

func length<T: Three>(_ vector: T) -> T.FloatType {
        return lengthSquared(vector).squareRoot()
}

func distanceSquared<T: Three>(_ left: T, _ right: T) -> T.FloatType {
        return lengthSquared(left - right)
}

func distance<T: Three>(_ left: T, _ right: T) -> T.FloatType {
        return distanceSquared(left, right).squareRoot()
}

prefix func - <T: Three>(vector: T) -> T {
        return T.init(x: -vector.x, y: -vector.y, z: -vector.z)
}

func / <T: Three>(vector: T, divisor: T.FloatType) -> T {
        return T.init(x: vector.x / divisor, y: vector.y / divisor, z: vector.z / divisor)
}

func / <T: Three>(vector: T, divisorVector: T) -> T {
        return T.init(x: vector.x / divisorVector.x, y: vector.y / divisorVector.y, z: vector.z / divisorVector.z)
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

public func normalized<T: Three>(_ vector: T) -> T {
        var ret: T
        let len = length(vector)
        if len == 0 {
                ret = vector
        } else {
                ret = vector / len
        }
        return ret
}

public func cross<T: Three>(_ vectorA: T, _ vectorB: T) -> T
where T.FloatType: FloatingPoint {

        // Ensure all arithmetic operations (multiplication, subtraction) use the correct FloatType
        let x = vectorA.y * vectorB.z - vectorA.z * vectorB.y
        let y = vectorA.z * vectorB.x - vectorA.x * vectorB.z
        let z = vectorA.x * vectorB.y - vectorA.y * vectorB.x

        // Create a new instance of the generic type T
        return T(x: x, y: y, z: z)
}

func dot<T: Three>(_ vectorA: T, _ vectorB: T) -> T.FloatType {
        return vectorA.x * vectorB.x + vectorA.y * vectorB.y + vectorA.z * vectorB.z
}

func absDot<T: Three>(_ vectorA: T, _ vectorB: T) -> T.FloatType {
        return abs(dot(vectorA, vectorB))
}
