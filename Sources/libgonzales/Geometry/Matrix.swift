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

        public static func * (left: Matrix, right: Matrix) -> Matrix {
                var result = Matrix()
                result.col0 =
                        left.col0 * right.col0.x + left.col1 * right.col0.y + left.col2 * right.col0.z + left
                        .col3 * right.col0.w
                result.col1 =
                        left.col0 * right.col1.x + left.col1 * right.col1.y + left.col2 * right.col1.z + left
                        .col3 * right.col1.w
                result.col2 =
                        left.col0 * right.col2.x + left.col1 * right.col2.y + left.col2 * right.col2.z + left
                        .col3 * right.col2.w
                result.col3 =
                        left.col0 * right.col3.x + left.col1 * right.col3.y + left.col2 * right.col3.z + left
                        .col3 * right.col3.w
                return result
        }

        public func invert(m _: Matrix) throws -> Matrix {

                var indxc = [0, 0, 0, 0]
                var indxr = [0, 0, 0, 0]
                var ipiv = [0, 0, 0, 0]
                var minv = self

                func choosePivot(irow: inout Int, icol: inout Int) throws {
                        var big: Real = 0.0
                        for row in 0..<4 where ipiv[row] != 1 {
                                for col in 0..<4 {
                                        if ipiv[col] == 0 {
                                                if abs(minv[row, col]) >= big {
                                                        big = abs(minv[row, col])
                                                        irow = row
                                                        icol = col
                                                }
                                        } else if ipiv[col] > 1 {
                                                throw MatrixError.singularMatrix
                                        }
                                }
                        }
                }

                func swapColumns() {
                        for colIndex in (0..<4).reversed() where indxr[colIndex] != indxc[colIndex] {
                                for rowIndex in 0..<4 {
                                        let rowSwap = indxr[colIndex]
                                        let colSwap = indxc[colIndex]
                                        let tmp = minv[rowIndex, rowSwap]
                                        minv[rowIndex, rowSwap] = minv[rowIndex, colSwap]
                                        minv[rowIndex, colSwap] = tmp
                                }
                        }
                }

                func subtractRow(icol: inout Int) {
                        for rowIndex in 0..<4 where rowIndex != icol {
                                let save = minv[rowIndex, icol]
                                minv[rowIndex, icol] = 0
                                for colIndex in 0..<4 {
                                        minv[rowIndex, colIndex] -= minv[icol, colIndex] * save
                                }
                        }
                }

                func reduce(_ iteration: Int) throws {
                        var irow = 0
                        var icol = 0
                        try choosePivot(irow: &irow, icol: &icol)
                        ipiv[icol] += 1
                        if irow != icol {
                                for colIndex in 0..<4 {
                                        let tmp = minv[irow, colIndex]
                                        minv[irow, colIndex] = minv[icol, colIndex]
                                        minv[icol, colIndex] = tmp
                                }
                        }
                        indxr[iteration] = irow
                        indxc[iteration] = icol
                        if minv[icol, icol] == 0 {
                                throw MatrixError.singularMatrix
                        }
                        let pivinv = 1 / minv[icol, icol]
                        minv[icol, icol] = 1
                        for colIndex in 0..<4 {
                                minv[icol, colIndex] *= pivinv
                        }
                        subtractRow(icol: &icol)
                }

                do {
                        for iteration in 0..<4 { try reduce(iteration) }
                        swapColumns()
                        return minv
                } catch MatrixError.singularMatrix {
                        if !hasPrintedSingularMatrixWarning {
                                print("Warning: Singular matrix encountered! \(self)")
                                hasPrintedSingularMatrixWarning = true
                        }
                        return Matrix()
                } catch {
                        throw error
                }
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
