public struct Matrix: Sendable {

        public init(
                t00: FloatX, t01: FloatX, t02: FloatX, t03: FloatX,
                t10: FloatX, t11: FloatX, t12: FloatX, t13: FloatX,
                t20: FloatX, t21: FloatX, t22: FloatX, t23: FloatX,
                t30: FloatX, t31: FloatX, t32: FloatX, t33: FloatX
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

        subscript(row: Int, column: Int) -> FloatX {
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

        public func invert(m _: Matrix) -> Matrix {

                var indxc = [0, 0, 0, 0]
                var indxr = [0, 0, 0, 0]
                var ipiv = [0, 0, 0, 0]
                var minv = backing

                func choosePivot(irow: inout Int, icol: inout Int) throws {
                        var big: FloatX = 0.0
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
                        warning("Singular matrix encountered!")
                        return Matrix()
                } catch {
                        abort("Unhandled error in matrix invert!")
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
                return invert(m: self)
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
