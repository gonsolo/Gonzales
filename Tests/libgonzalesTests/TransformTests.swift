import Testing

@testable import libgonzales

@Suite struct TransformTests {

        // MARK: - Identity Transform

        @Test func identityPreservesVector() {
                let t = Transform()
                let v = Vector(x: 1, y: 2, z: 3)
                let result = t * v
                #expect(abs(result.x - v.x) <= 1e-6)
                #expect(abs(result.y - v.y) <= 1e-6)
                #expect(abs(result.z - v.z) <= 1e-6)
        }

        @Test func identityPreservesPoint() {
                let t = Transform()
                let p = Point(x: 4, y: 5, z: 6)
                let result = t * p
                #expect(abs(result.x - p.x) <= 1e-6)
                #expect(abs(result.y - p.y) <= 1e-6)
                #expect(abs(result.z - p.z) <= 1e-6)
        }

        @Test func identityPreservesNormal() {
                let t = Transform()
                let n = Normal(x: 0, y: 1, z: 0)
                let result = t * n
                #expect(abs(result.x - n.x) <= 1e-6)
                #expect(abs(result.y - n.y) <= 1e-6)
                #expect(abs(result.z - n.z) <= 1e-6)
        }

        // MARK: - Scale Transform

        @Test func scaleTransformVector() throws {
                let t = try Transform.makeScale(x: 2, y: 3, z: 4)
                let v = Vector(x: 1, y: 1, z: 1)
                let result = t * v
                #expect(abs(result.x - 2) <= 1e-6)
                #expect(abs(result.y - 3) <= 1e-6)
                #expect(abs(result.z - 4) <= 1e-6)
        }

        @Test func scaleTransformPoint() throws {
                let t = try Transform.makeScale(x: 2, y: 3, z: 4)
                let p = Point(x: 1, y: 1, z: 1)
                let result = t * p
                #expect(abs(result.x - 2) <= 1e-6)
                #expect(abs(result.y - 3) <= 1e-6)
                #expect(abs(result.z - 4) <= 1e-6)
        }

        // MARK: - Translation Transform

        @Test func translationTransformPoint() throws {
                let t = try Transform.makeTranslation(from: Vector(x: 10, y: 20, z: 30))
                let p = Point(x: 1, y: 2, z: 3)
                let result = t * p
                #expect(abs(result.x - 11) <= 1e-6)
                #expect(abs(result.y - 22) <= 1e-6)
                #expect(abs(result.z - 33) <= 1e-6)
        }

        @Test func translationDoesNotAffectVector() throws {
                let t = try Transform.makeTranslation(from: Vector(x: 10, y: 20, z: 30))
                let v = Vector(x: 1, y: 2, z: 3)
                let result = t * v
                // Vectors are translation-invariant
                #expect(abs(result.x - 1) <= 1e-6)
                #expect(abs(result.y - 2) <= 1e-6)
                #expect(abs(result.z - 3) <= 1e-6)
        }

        // MARK: - Transform Composition

        @Test func compositionScaleThenTranslate() throws {
                let scale = try Transform.makeScale(x: 2, y: 2, z: 2)
                let translate = try Transform.makeTranslation(from: Vector(x: 1, y: 0, z: 0))
                // translate * scale: first scale, then translate
                let combined = try translate * scale
                let p = Point(x: 1, y: 0, z: 0)
                let result = combined * p
                // Scale (1,0,0) -> (2,0,0), then translate -> (3,0,0)
                #expect(abs(result.x - 3) <= 1e-6)
                #expect(abs(result.y) <= 1e-6)
                #expect(abs(result.z) <= 1e-6)
        }

        // MARK: - Inverse

        @Test func inverseRoundTrip() throws {
                let t = try Transform.makeScale(x: 2, y: 3, z: 4)
                let p = Point(x: 1, y: 2, z: 3)
                let transformed = t * p
                let recovered = t.inverse * transformed
                #expect(abs(recovered.x - p.x) <= 1e-4)
                #expect(abs(recovered.y - p.y) <= 1e-4)
                #expect(abs(recovered.z - p.z) <= 1e-4)
        }
}
