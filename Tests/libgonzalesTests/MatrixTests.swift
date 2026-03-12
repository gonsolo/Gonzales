import Testing

@testable import libgonzales

@Suite struct MatrixTests {

        // MARK: - Default Init (Identity)

        @Test func defaultIsIdentity() {
                let m = Matrix()
                for i in 0..<4 {
                        for j in 0..<4 {
                                let expected: Real = (i == j) ? 1 : 0
                                #expect(abs(m[i, j] - expected) <= 1e-6)
                        }
                }
        }

        // MARK: - Subscript

        @Test func subscriptSetGet() {
                var m = Matrix()
                m[2, 3] = 42
                #expect(abs(m[2, 3] - 42) <= 1e-6)
        }

        // MARK: - Multiplication

        @Test func multiplyByIdentity() {
                let identity = Matrix()
                let m = Matrix(
                        t00: 1, t01: 2, t02: 3, t03: 4,
                        t10: 5, t11: 6, t12: 7, t13: 8,
                        t20: 9, t21: 10, t22: 11, t23: 12,
                        t30: 13, t31: 14, t32: 15, t33: 16)
                let result = identity * m
                for i in 0..<4 {
                        for j in 0..<4 {
                                #expect(abs(result[i, j] - m[i, j]) <= 1e-6)
                        }
                }
        }

        @Test func multiplyKnownResult() {
                // Diagonal matrices: diag(2,3,4,1) * diag(5,6,7,1) = diag(10,18,28,1)
                let a = Matrix(
                        t00: 2, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: 3, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: 4, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let b = Matrix(
                        t00: 5, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: 6, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: 7, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let result = a * b
                #expect(abs(result[0, 0] - 10) <= 1e-6)
                #expect(abs(result[1, 1] - 18) <= 1e-6)
                #expect(abs(result[2, 2] - 28) <= 1e-6)
                #expect(abs(result[3, 3] - 1) <= 1e-6)
                // Off-diagonals should be zero
                #expect(abs(result[0, 1]) <= 1e-6)
                #expect(abs(result[1, 0]) <= 1e-6)
        }

        // MARK: - Transpose

        @Test func transposeSwapsRowsAndColumns() {
                let m = Matrix(
                        t00: 1, t01: 2, t02: 3, t03: 4,
                        t10: 5, t11: 6, t12: 7, t13: 8,
                        t20: 9, t21: 10, t22: 11, t23: 12,
                        t30: 13, t31: 14, t32: 15, t33: 16)
                let t = m.transpose()
                #expect(abs(t[0, 1] - 5) <= 1e-6)
                #expect(abs(t[1, 0] - 2) <= 1e-6)
                #expect(abs(t[2, 3] - 15) <= 1e-6)
                #expect(abs(t[3, 2] - 12) <= 1e-6)
        }

        @Test func doubleTransposeIsOriginal() {
                let m = Matrix(
                        t00: 1, t01: 2, t02: 3, t03: 4,
                        t10: 5, t11: 6, t12: 7, t13: 8,
                        t20: 9, t21: 10, t22: 11, t23: 12,
                        t30: 13, t31: 14, t32: 15, t33: 16)
                let result = m.transpose().transpose()
                for i in 0..<4 {
                        for j in 0..<4 {
                                #expect(abs(result[i, j] - m[i, j]) <= 1e-6)
                        }
                }
        }

        // MARK: - Inverse

        @Test func inverseOfDiagonal() throws {
                let m = Matrix(
                        t00: 2, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: 4, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: 5, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let inv = try m.inverse
                #expect(abs(inv[0, 0] - 0.5) <= 1e-5)
                #expect(abs(inv[1, 1] - 0.25) <= 1e-5)
                #expect(abs(inv[2, 2] - 0.2) <= 1e-5)
                #expect(abs(inv[3, 3] - 1) <= 1e-5)
        }

        @Test func inverseTimesOriginalIsIdentity() throws {
                let m = Matrix(
                        t00: 2, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: 4, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: 5, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let result = m * (try m.inverse)
                for i in 0..<4 {
                        for j in 0..<4 {
                                let expected: Real = (i == j) ? 1 : 0
                                #expect(abs(result[i, j] - expected) <= 1e-4)
                        }
                }
        }
}
