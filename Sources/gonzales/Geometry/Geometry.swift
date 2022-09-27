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
        let vx = sinTheta * cos(phi) * x
        let vy = sinTheta * sin(phi) * y
        let vz = cosTheta * z
        return vx + vy + vz
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

func maxDimension<T: Comparable>(_ vector: Vector3<T>) -> Int {
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

func makeCoordinateSystem(from v1: Vector) -> (v2: Vector, v3: Vector) {
        var v2 = up
        if abs(v1.x) > abs(v1.y) {
                v2 = Vector(x: -v1.z, y: 0, z: v1.x) / (v1.x * v1.x + v1.z * v1.z).squareRoot()
        } else {
                v2 = Vector(x: 0, y: v1.z, z: -v1.y) / (v1.y * v1.y + v1.z * v1.z).squareRoot()
        }
        let v3 = cross(v1, v2)
        return (v2, v3)
}

func faceforward(normal: Normal, comparedTo vector: Vector) -> Normal {
        return dot(normal, vector) < 0 ? -normal : normal
}
