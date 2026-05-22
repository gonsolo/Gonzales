import mojoKernel

private nonisolated(unsafe) var hasPrintedSingularMatrixWarning = false

public struct Matrix: Sendable {

        public var col0: SIMD4<Real>
        public var col1: SIMD4<Real>
        public var col2: SIMD4<Real>
        public var col3: SIMD4<Real>

        public init(
                t00: Real, t01: Real, t02: Real, t03: Real,
                t10: Real, t11: Real, t12: Real, t13: Real,
                t20: Real, t21: Real, t22: Real, t23: Real,
                t30: Real, t31: Real, t32: Real, t33: Real
        ) {
                self.col0 = SIMD4<Real>(t00, t10, t20, t30)
                self.col1 = SIMD4<Real>(t01, t11, t21, t31)
                self.col2 = SIMD4<Real>(t02, t12, t22, t32)
                self.col3 = SIMD4<Real>(t03, t13, t23, t33)
        }

        public init(matrix: Matrix) {
                self.col0 = matrix.col0
                self.col1 = matrix.col1
                self.col2 = matrix.col2
                self.col3 = matrix.col3
        }

        public init() {
                self.col0 = SIMD4<Real>(1, 0, 0, 0)
                self.col1 = SIMD4<Real>(0, 1, 0, 0)
                self.col2 = SIMD4<Real>(0, 0, 1, 0)
                self.col3 = SIMD4<Real>(0, 0, 0, 1)
        }

        subscript(row: Int, column: Int) -> Real {
                get {
                        switch column {
                        case 0: return col0[row]
                        case 1: return col1[row]
                        case 2: return col2[row]
                        case 3: return col3[row]
                        default: fatalError("Matrix column index out of bounds")
                        }
                }
                set {
                        switch column {
                        case 0: col0[row] = newValue
                        case 1: col1[row] = newValue
                        case 2: col2[row] = newValue
                        case 3: col3[row] = newValue
                        default: fatalError("Matrix column index out of bounds")
                        }
                }
        }

        // Flat 16-float column-major view (col0.x..col0.w, col1..., col2..., col3...),
        // the layout the Mojo matrix kernels operate on.
        func columnMajorFloats() -> [Float] {
                return [
                        col0.x, col0.y, col0.z, col0.w,
                        col1.x, col1.y, col1.z, col1.w,
                        col2.x, col2.y, col2.z, col2.w,
                        col3.x, col3.y, col3.z, col3.w,
                ]
        }

        init(columnMajorFloats f: [Float]) {
                col0 = SIMD4<Real>(f[0], f[1], f[2], f[3])
                col1 = SIMD4<Real>(f[4], f[5], f[6], f[7])
                col2 = SIMD4<Real>(f[8], f[9], f[10], f[11])
                col3 = SIMD4<Real>(f[12], f[13], f[14], f[15])
        }

        public static func * (left: Matrix, right: Matrix) -> Matrix {
                let a = left.columnMajorFloats()
                let b = right.columnMajorFloats()
                var out = [Float](repeating: 0, count: 16)
                a.withUnsafeBufferPointer { ap in
                        b.withUnsafeBufferPointer { bp in
                                out.withUnsafeMutableBufferPointer { op in
                                        mojo_matrix_multiply(ap.baseAddress, bp.baseAddress, op.baseAddress)
                                }
                        }
                }
                return Matrix(columnMajorFloats: out)
        }

        public func invert(m _: Matrix) throws -> Matrix {
                let input = columnMajorFloats()
                var out = [Float](repeating: 0, count: 16)
                let ok = input.withUnsafeBufferPointer { ip in
                        out.withUnsafeMutableBufferPointer { op in
                                mojo_matrix_invert(ip.baseAddress, op.baseAddress)
                        }
                }
                if ok == 0 && !hasPrintedSingularMatrixWarning {
                        print("Warning: Singular matrix encountered! \(self)")
                        hasPrintedSingularMatrixWarning = true
                }
                return Matrix(columnMajorFloats: out)
        }

        func transpose() -> Matrix {
                return Matrix(
                        t00: col0.x, t01: col0.y, t02: col0.z, t03: col0.w,
                        t10: col1.x, t11: col1.y, t12: col1.z, t13: col1.w,
                        t20: col2.x, t21: col2.y, t22: col2.z, t23: col2.w,
                        t30: col3.x, t31: col3.y, t32: col3.z, t33: col3.w
                )
        }

        public var inverse: Matrix {
                get throws {
                        return try invert(m: self)
                }
        }
}

extension Matrix: CustomStringConvertible {
        public var description: String {
                var desc = ""
                for rowIndex in 0..<4 {
                        desc += "[ "
                        for columnIndex in 0..<4 {
                                desc += "\(self[rowIndex, columnIndex])"
                                if columnIndex != 3 { desc += " " }
                        }
                        desc += " ]"
                }
                return desc
        }
}
