import Foundation  // tan

struct Transform: Sendable {

        public init(matrix: Matrix, inverseMatrix: Matrix) {
                self.matrix = matrix
                self.inverseMatrix = inverseMatrix
        }

        public init(matrix: Matrix) {
                self.init(matrix: matrix, inverseMatrix: matrix.inverse)
        }

        public init() {
                matrix = Matrix()
                inverseMatrix = Matrix()
        }

        // For Embree
        public func getMatrix() -> Matrix {
                return matrix
        }

        var inverse: Transform {
                return Transform(
                        matrix: self.inverseMatrix, inverseMatrix: self.matrix)
        }

        private var matrix: Matrix
        private var inverseMatrix: Matrix
}

extension Transform {

        @inline(__always)
        public static func * (transform: Transform, vector: Vector) -> Vector {
                let matrix = transform.matrix
                return Vector(
                        x: matrix[0, 0] * vector.x + matrix[0, 1] * vector.y + matrix[0, 2] * vector.z,
                        y: matrix[1, 0] * vector.x + matrix[1, 1] * vector.y + matrix[1, 2] * vector.z,
                        z: matrix[2, 0] * vector.x + matrix[2, 1] * vector.y + matrix[2, 2] * vector.z)
        }

        @inline(__always)
        public static func * (transform: Transform, normalVector: Normal) -> Normal {
                let inverseMatrix = transform.inverse.matrix
                return Normal(
                        x: inverseMatrix[0, 0] * normalVector.x + inverseMatrix[1, 0] * normalVector.y
                                + inverseMatrix[2, 0] * normalVector.z,
                        y: inverseMatrix[0, 1] * normalVector.x + inverseMatrix[1, 1] * normalVector.y
                                + inverseMatrix[2, 1] * normalVector.z,
                        z: inverseMatrix[0, 2] * normalVector.x + inverseMatrix[1, 2] * normalVector.y
                                + inverseMatrix[2, 2] * normalVector.z)
        }

        @inline(__always)
        public static func * (transform: Transform, point: Point) -> Point {
                let matrix = transform.matrix
                let pointLocal = Point(
                        x: matrix[0, 0] * point.x + matrix[0, 1] * point.y + matrix[0, 2] * point.z + matrix[0, 3],
                        y: matrix[1, 0] * point.x + matrix[1, 1] * point.y + matrix[1, 2] * point.z + matrix[1, 3],
                        z: matrix[2, 0] * point.x + matrix[2, 1] * point.y + matrix[2, 2] * point.z + matrix[2, 3])
                let weightP = matrix[3, 0] * point.x + matrix[3, 1] * point.y + matrix[3, 2] * point.z + matrix[3, 3]
                return pointLocal / weightP
        }

        public static func *= (left: inout Transform, right: Transform) {
                left = left * right
        }

        public static func * (left: Transform, right: Transform) -> Transform {
                return Transform(matrix: left.matrix * right.matrix)
        }
}

extension Transform: CustomStringConvertible {
        public var description: String {
                return "Transform:\n" + matrix.description + "\n"
        }
}

extension Transform {
        static public func makePerspective(fov: FloatX, near: FloatX, far _: FloatX) throws
                -> Transform {
                let persp = Matrix(
                        t00: 1, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: 1, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: fov / (fov - near), t23: -fov * near / (fov - near),
                        t30: 0, t31: 0, t32: 1, t33: 0)
                let invTanAng = 1 / tan(radians(deg: fov) / 2)
                return try makeScale(x: invTanAng, y: invTanAng, z: 1) * Transform(matrix: persp)
        }

        static func makeScale(x: FloatX, y: FloatX, z: FloatX) throws -> Transform {
                let matrix = Matrix(
                        t00: x, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: y, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: z, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                return Transform(matrix: matrix)
        }

        static func makeTranslation(from delta: Vector) throws -> Transform {
                let matrix = Matrix(
                        t00: 1, t01: 0, t02: 0, t03: delta.x,
                        t10: 0, t11: 1, t12: 0, t13: delta.y,
                        t20: 0, t21: 0, t22: 1, t23: delta.z,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                return Transform(matrix: matrix)
        }
}
