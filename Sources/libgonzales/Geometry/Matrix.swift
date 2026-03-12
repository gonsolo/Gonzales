public struct Matrix: Sendable {

        public init(
                t00: Real, t01: Real, t02: Real, t03: Real,
                t10: Real, t11: Real, t12: Real, t13: Real,
                t20: Real, t21: Real, t22: Real, t23: Real,
                t30: Real, t31: Real, t32: Real, t33: Real
        ) {
                self.init()

                backing[0, 0] = t00
                backing[0, 1] = t01
                backing[0, 2] = t02
                backing[0, 3] = t03

                backing[1, 0] = t10
                backing[1, 1] = t11
                backing[1, 2] = t12
                backing[1, 3] = t13

                backing[2, 0] = t20
                backing[2, 1] = t21
                backing[2, 2] = t22
                backing[2, 3] = t23

                backing[3, 0] = t30
                backing[3, 1] = t31
                backing[3, 2] = t32
                backing[3, 3] = t33
        }

        init(backing: MatrixBacking) {
                self.backing = backing
        }

        public init(matrix: Matrix) {
                self.backing = matrix.backing
        }

        public init() {
                backing = MatrixBacking()
        }

        subscript(row: Int, column: Int) -> Real {
                get { return backing[row, column] }
                set { backing[row, column] = newValue }
        }

        public static func * (matrix1: Matrix, matrix2: Matrix) -> Matrix {
                var result = Matrix()
                for rowIndex in 0..<4 {
                        for columnIndex in 0..<4 {
                                let valA = matrix1[rowIndex, 0] * matrix2[0, columnIndex]
                                let valB = matrix1[rowIndex, 1] * matrix2[1, columnIndex]
                                let valC = matrix1[rowIndex, 2] * matrix2[2, columnIndex]
                                let valD = matrix1[rowIndex, 3] * matrix2[3, columnIndex]
                                result[rowIndex, columnIndex] = valA + valB + valC + valD
                        }
                }
                return result
        }

        public func invert(m _: Matrix) throws -> Matrix {

                var indxc = [0, 0, 0, 0]
                var indxr = [0, 0, 0, 0]
                var ipiv = [0, 0, 0, 0]
                var minv = backing

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
                        // return Matrix(minv)
                        return Matrix(backing: minv)
                } catch MatrixError.singularMatrix {
                        print("Warning: Singular matrix encountered!")
                        return Matrix()
                } catch {
                        throw error
                }
        }

        func transpose() -> Matrix {
                var transposed = Matrix()
                for rowIndex in 0..<4 {
                        for columnIndex in 0..<4 {
                                transposed[rowIndex, columnIndex] = backing[columnIndex, rowIndex]
                        }
                }
                return transposed
        }

        public var inverse: Matrix {
                get throws {
                        return try invert(m: self)
                }
        }

        var backing: MatrixBacking
}

extension Matrix: CustomStringConvertible {
        public var description: String {
                var desc = ""
                for rowIndex in 0..<4 {
                        desc += "[ "
                        for columnIndex in 0..<4 {
                                desc += "\(backing[rowIndex, columnIndex])"
                                if columnIndex != 3 { desc += " " }
                        }
                        desc += " ]"
                }
                return desc
        }
}
