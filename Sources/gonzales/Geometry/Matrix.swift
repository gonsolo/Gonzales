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

        public init(m: Matrix) {
                self.backing = m.backing
        }

        public init() {
                backing = MatrixBacking()
        }

        subscript(row: Int, column: Int) -> FloatX {
                get { return backing[row, column] }
                set { backing[row, column] = newValue }
        }

        public static func * (m1: Matrix, m2: Matrix) -> Matrix {
                var r = Matrix()
                for i in 0..<4 {
                        for j in 0..<4 {
                                let a = m1[i, 0] * m2[0, j]
                                let b = m1[i, 1] * m2[1, j]
                                let c = m1[i, 2] * m2[2, j]
                                let d = m1[i, 3] * m2[3, j]
                                r[i, j] = a + b + c + d
                        }
                }
                return r
        }

        @MainActor
        public func invert(m: Matrix) -> Matrix {

                var indxc = [0, 0, 0, 0]
                var indxr = [0, 0, 0, 0]
                var ipiv = [0, 0, 0, 0]
                var minv = backing

                func choosePivot(irow: inout Int, icol: inout Int) throws {
                        var big: FloatX = 0.0
                        for j in 0..<4 {
                                if ipiv[j] != 1 {
                                        for k in 0..<4 {
                                                if ipiv[k] == 0 {
                                                        if abs(minv[j, k]) >= big {
                                                                big = abs(minv[j, k])
                                                                irow = j
                                                                icol = k
                                                        }
                                                } else if ipiv[k] > 1 {
                                                        throw MatrixError.singularMatrix
                                                }
                                        }
                                }
                        }
                }

                func swapColumns() {
                        for j in (0..<4).reversed() {
                                if indxr[j] != indxc[j] {
                                        for k in 0..<4 {
                                                let r = indxr[j]
                                                let c = indxc[j]
                                                let tmp = minv[k, r]
                                                minv[k, r] = minv[k, c]
                                                minv[k, c] = tmp
                                        }
                                }
                        }
                }

                func subtractRow(icol: inout Int) {
                        for j in 0..<4 {
                                if j != icol {
                                        let save = minv[j, icol]
                                        minv[j, icol] = 0
                                        for k in 0..<4 {
                                                minv[j, k] -= minv[icol, k] * save
                                        }
                                }
                        }
                }

                func reduce(_ i: Int) throws {
                        var irow = 0
                        var icol = 0
                        try choosePivot(irow: &irow, icol: &icol)
                        ipiv[icol] = ipiv[icol] + 1
                        if irow != icol {
                                for k in 0..<4 {
                                        let tmp = minv[irow, k]
                                        minv[irow, k] = minv[icol, k]
                                        minv[icol, k] = tmp
                                }
                        }
                        indxr[i] = irow
                        indxc[i] = icol
                        if minv[icol, icol] == 0 {
                                throw MatrixError.singularMatrix
                        }
                        let pivinv = 1 / minv[icol, icol]
                        minv[icol, icol] = 1
                        for j in 0..<4 {
                                minv[icol, j] *= pivinv
                        }
                        subtractRow(icol: &icol)
                }

                do {
                        for i in 0..<4 { try reduce(i) }
                        swapColumns()
                        //return Matrix(minv)
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
                for x in 0..<4 {
                        for y in 0..<4 {
                                transposed[x, y] = backing[y, x]
                        }
                }
                return transposed
        }

        @MainActor
        public var inverse: Matrix {
                return invert(m: self)
        }

        var backing: MatrixBacking
}

extension Matrix: CustomStringConvertible {
        public var description: String {
                var d = ""
                for i in 0..<4 {
                        d += "[ "
                        for j in 0..<4 {
                                d += "\(backing[i, j])"
                                if j != 3 { d += " " }
                        }
                        d += " ]"
                }
                return d
        }
}
