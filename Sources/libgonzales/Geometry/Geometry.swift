import Foundation  // sin, cos

func sphericalDirection(sinTheta: FloatX, cosTheta: FloatX, phi: FloatX) -> Vector {
        return Vector(x: sinTheta * cos(phi), y: sinTheta * sin(phi), z: cosTheta)
}

func sphericalDirection(
        sinTheta: FloatX,
        cosTheta: FloatX,
        phi: FloatX,
        x: Vector,
        y: Vector,
        z: Vector
) -> Vector {
        let vectorX = sinTheta * cos(phi) * x
        let vectorY = sinTheta * sin(phi) * y
        let vectorZ = cosTheta * z
        return vectorX + vectorY + vectorZ
}

func sphericalCoordinatesFrom(vector: Vector) -> (theta: FloatX, phi: FloatX) {
        let theta = acos(vector.z)
        let atan = atan2(vector.y, vector.x)
        let phi = atan < 0 ? atan + 2 * FloatX.pi : atan
        return (theta, phi)
}

func dot(_ a: Vector, _ b: Normal) -> FloatX {
        return a.x * b.x + a.y * b.y + a.z * b.z
}

func dot(_ a: Normal, _ b: Vector) -> FloatX {
        return a.x * b.x + a.y * b.y + a.z * b.z
}

func dot(_ a: Vector2F, _ b: Vector2F) -> FloatX {
        return a.x * b.x + a.y * b.y
}

func absDot(_ a: Vector, _ b: Normal) -> FloatX {
        return abs(dot(a, b))
}

func absDot(_ a: Normal, _ b: Vector) -> FloatX {
        return abs(dot(Vector(normal: a), b))
}

func maxDimension(_ vector: Vector3) -> Int {
        return (vector.x > vector.y)
                ? ((vector.x > vector.z) ? 0 : 2) : ((vector.y > vector.z) ? 1 : 2)
}

func union(bound: Bounds3f, point: Point) -> Bounds3f {
        let pMin = Point(
                x: min(bound.pMin.x, point.x),
                y: min(bound.pMin.y, point.y),
                z: min(bound.pMin.z, point.z))
        let pMax = Point(
                x: max(bound.pMax.x, point.x),
                y: max(bound.pMax.y, point.y),
                z: max(bound.pMax.z, point.z))
        return Bounds3f(first: pMin, second: pMax)
}

@inline(__always)
func makeCoordinateSystem(from vector1: Vector) -> (vector2: Vector, vector3: Vector) {
        var vector2 = upVector
        if abs(vector1.x) > abs(vector1.y) {
                vector2 = Vector(x: -vector1.z, y: 0, z: vector1.x) / (vector1.x * vector1.x + vector1.z * vector1.z).squareRoot()
        } else {
                vector2 = Vector(x: 0, y: vector1.z, z: -vector1.y) / (vector1.y * vector1.y + vector1.z * vector1.z).squareRoot()
        }
        let vector3 = cross(vector1, vector2)
        return (vector2, vector3)
}

func faceforward(normal: Normal, comparedTo vector: Vector) -> Normal {
        return dot(normal, vector) < 0 ? -normal : normal
}

func faceforward(vector: Vector, comparedTo normal: Normal) -> Vector {
        return dot(vector, normal) < 0 ? -vector : vector
}
