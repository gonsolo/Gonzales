import Testing

@testable import libgonzales

@Suite struct VectorTests {

        // MARK: - Construction

        @Test func defaultConstruction() {
                let v = Vector3()
                #expect(abs(v.x) <= 1e-6)
                #expect(abs(v.y) <= 1e-6)
                #expect(abs(v.z - 1) <= 1e-6)
        }

        @Test func valueConstruction() {
                let v = Vector3(value: 5)
                #expect(abs(v.x - 5) <= 1e-6)
                #expect(abs(v.y - 5) <= 1e-6)
                #expect(abs(v.z - 5) <= 1e-6)
        }

        // MARK: - Arithmetic

        @Test func addition() {
                let a = Vector(x: 1, y: 2, z: 3)
                let b = Vector(x: 4, y: 5, z: 6)
                let result = a + b
                #expect(abs(result.x - 5) <= 1e-6)
                #expect(abs(result.y - 7) <= 1e-6)
                #expect(abs(result.z - 9) <= 1e-6)
        }

        @Test func subtraction() {
                let a = Vector(x: 5, y: 7, z: 9)
                let b = Vector(x: 1, y: 2, z: 3)
                let result = a - b
                #expect(abs(result.x - 4) <= 1e-6)
                #expect(abs(result.y - 5) <= 1e-6)
                #expect(abs(result.z - 6) <= 1e-6)
        }

        @Test func scalarMultiplication() {
                let v = Vector(x: 1, y: 2, z: 3)
                let result = v * 3
                #expect(abs(result.x - 3) <= 1e-6)
                #expect(abs(result.y - 6) <= 1e-6)
                #expect(abs(result.z - 9) <= 1e-6)
        }

        @Test func scalarMultiplicationCommutative() {
                let v = Vector(x: 1, y: 2, z: 3)
                let left = 3 * v
                let right = v * 3
                #expect(abs(left.x - right.x) <= 1e-6)
                #expect(abs(left.y - right.y) <= 1e-6)
                #expect(abs(left.z - right.z) <= 1e-6)
        }

        @Test func scalarDivision() {
                let v = Vector(x: 6, y: 9, z: 12)
                let result = v / 3
                #expect(abs(result.x - 2) <= 1e-6)
                #expect(abs(result.y - 3) <= 1e-6)
                #expect(abs(result.z - 4) <= 1e-6)
        }

        @Test func negation() {
                let v = Vector(x: 1, y: -2, z: 3)
                let result = -v
                #expect(abs(result.x - (-1)) <= 1e-6)
                #expect(abs(result.y - 2) <= 1e-6)
                #expect(abs(result.z - (-3)) <= 1e-6)
        }

        // MARK: - Compound Assignment

        @Test func addAssign() {
                var v = Vector(x: 1, y: 2, z: 3)
                v += Vector(x: 4, y: 5, z: 6)
                #expect(abs(v.x - 5) <= 1e-6)
                #expect(abs(v.y - 7) <= 1e-6)
                #expect(abs(v.z - 9) <= 1e-6)
        }

        @Test func subtractAssign() {
                var v = Vector(x: 5, y: 7, z: 9)
                v -= Vector(x: 1, y: 2, z: 3)
                #expect(abs(v.x - 4) <= 1e-6)
                #expect(abs(v.y - 5) <= 1e-6)
                #expect(abs(v.z - 6) <= 1e-6)
        }

        @Test func multiplyAssign() {
                var v = Vector(x: 1, y: 2, z: 3)
                v *= 3
                #expect(abs(v.x - 3) <= 1e-6)
                #expect(abs(v.y - 6) <= 1e-6)
                #expect(abs(v.z - 9) <= 1e-6)
        }

        @Test func divideAssign() {
                var v = Vector(x: 6, y: 9, z: 12)
                v /= 3
                #expect(abs(v.x - 2) <= 1e-6)
                #expect(abs(v.y - 3) <= 1e-6)
                #expect(abs(v.z - 4) <= 1e-6)
        }

        // MARK: - Utilities

        @Test func mirrorFlipsXYPreservesZ() {
                let v = Vector(x: 1, y: 2, z: 3)
                let result = mirror(v)
                #expect(abs(result.x - (-1)) <= 1e-6)
                #expect(abs(result.y - (-2)) <= 1e-6)
                #expect(abs(result.z - 3) <= 1e-6)
        }

        @Test func permuteReordersComponents() {
                let v = Vector(x: 10, y: 20, z: 30)
                let result = permute(vector: v, x: 2, y: 0, z: 1)
                #expect(abs(result.x - 30) <= 1e-6)
                #expect(abs(result.y - 10) <= 1e-6)
                #expect(abs(result.z - 20) <= 1e-6)
        }

        @Test func isNaNDetectsNaN() {
                let normal = Vector(x: 1, y: 2, z: 3)
                let nanVec = Vector(x: Real.nan, y: 0, z: 0)
                #expect(!normal.isNaN)
                #expect(nanVec.isNaN)
        }

        @Test func isZeroDetectsZero() {
                let zero = Vector(x: 0, y: 0, z: 0)
                let nonZero = Vector(x: 1, y: 0, z: 0)
                #expect(zero.isZero)
                #expect(!nonZero.isZero)
        }

        @Test func subscriptAccess() {
                let v = Vector(x: 10, y: 20, z: 30)
                #expect(abs(v[0] - 10) <= 1e-6)
                #expect(abs(v[1] - 20) <= 1e-6)
                #expect(abs(v[2] - 30) <= 1e-6)
        }

        @Test func subscriptSet() {
                var v = Vector(x: 0, y: 0, z: 0)
                v[0] = 10
                v[1] = 20
                v[2] = 30
                #expect(abs(v.x - 10) <= 1e-6)
                #expect(abs(v.y - 20) <= 1e-6)
                #expect(abs(v.z - 30) <= 1e-6)
        }

        // MARK: - Cross Product and Length

        @Test func crossProductOrthogonal() {
                let a = Vector(x: 1, y: 0, z: 0)
                let b = Vector(x: 0, y: 1, z: 0)
                let result = cross(a, b)
                #expect(abs(result.x) <= 1e-6)
                #expect(abs(result.y) <= 1e-6)
                #expect(abs(result.z - 1) <= 1e-6)
        }

        @Test func crossProductAntiCommutative() {
                let a = Vector(x: 1, y: 2, z: 3)
                let b = Vector(x: 4, y: 5, z: 6)
                let ab = cross(a, b)
                let ba = cross(b, a)
                #expect(abs(ab.x - (-ba.x)) <= 1e-6)
                #expect(abs(ab.y - (-ba.y)) <= 1e-6)
                #expect(abs(ab.z - (-ba.z)) <= 1e-6)
        }

        @Test func lengthUnitVector() {
                let v = Vector(x: 1, y: 0, z: 0)
                #expect(abs(length(v) - 1.0) <= 1e-6)
        }

        @Test func lengthKnown() {
                // length of (3,4,0) = 5
                let v = Vector(x: 3, y: 4, z: 0)
                #expect(abs(length(v) - 5.0) <= 1e-6)
        }

        @Test func normalizedHasUnitLength() {
                let v = Vector(x: 3, y: 4, z: 5)
                let n = normalized(v)
                #expect(abs(length(n) - 1.0) <= 1e-5)
        }
}
