import Glibc
import Testing

@testable import libgonzales

@Suite struct GeometryTests {

        // MARK: - Dot Product

        @Test func dotVectorNormalOrthogonal() {
                let vector = Vector(x: 1, y: 0, z: 0)
                let normal = Normal(x: 0, y: 1, z: 0)
                #expect(abs(dot(vector, normal)) <= 1e-6)
        }

        @Test func dotVectorNormalParallel() {
                let vector = Vector(x: 0, y: 0, z: 3)
                let normal = Normal(x: 0, y: 0, z: 2)
                #expect(abs(dot(vector, normal) - 6.0) <= 1e-6)
        }

        @Test func dotNormalVectorCommutes() {
                let vector = Vector(x: 1, y: 2, z: 3)
                let normal = Normal(x: 4, y: 5, z: 6)
                #expect(abs(dot(vector, normal) - dot(normal, vector)) <= 1e-6)
        }

        // MARK: - AbsDot

        @Test func absDotAlwaysNonNegative() {
                let vector = Vector(x: 1, y: 0, z: 0)
                let normal = Normal(x: -1, y: 0, z: 0)
                let result = absDot(vector, normal)
                #expect(result >= 0)
                #expect(abs(result - 1.0) <= 1e-6)
        }

        // MARK: - Faceforward

        @Test func faceforwardNormalFlips() {
                let normal = Normal(x: 0, y: 0, z: -1)
                let reference = Vector(x: 0, y: 0, z: 1)
                let result = faceForward(normal: normal, comparedTo: reference)
                // dot(normal, reference) < 0, so should flip
                #expect(abs(result.z - 1.0) <= 1e-6)
        }

        @Test func faceforwardNormalNoFlip() {
                let normal = Normal(x: 0, y: 0, z: 1)
                let reference = Vector(x: 0, y: 0, z: 1)
                let result = faceForward(normal: normal, comparedTo: reference)
                #expect(abs(result.z - 1.0) <= 1e-6)
        }

        @Test func faceforwardVectorFlips() {
                let vector = Vector(x: 0, y: 0, z: -1)
                let normal = Normal(x: 0, y: 0, z: 1)
                let result = faceForward(vector: vector, comparedTo: normal)
                #expect(abs(result.z - 1.0) <= 1e-6)
        }

        // MARK: - MakeCoordinateSystem

        @Test func makeCoordinateSystemOrthonormal() {
                let v1 = normalized(Vector(x: 1, y: 1, z: 0))
                let (v2, v3) = makeCoordinateSystem(from: v1)

                // Orthogonality
                #expect(abs(dot(v1, v2)) <= 1e-5)
                #expect(abs(dot(v1, v3)) <= 1e-5)
                #expect(abs(dot(v2, v3)) <= 1e-5)

                // Unit length
                #expect(abs(length(v2) - 1.0) <= 1e-5)
                #expect(abs(length(v3) - 1.0) <= 1e-5)
        }

        @Test func makeCoordinateSystemFromZAxis() {
                let v1 = Vector(x: 0, y: 0, z: 1)
                let (v2, v3) = makeCoordinateSystem(from: v1)

                #expect(abs(dot(v1, v2)) <= 1e-5)
                #expect(abs(dot(v1, v3)) <= 1e-5)
                #expect(abs(dot(v2, v3)) <= 1e-5)
        }

        // MARK: - Spherical Coordinates

        @Test func sphericalDirectionAtPoles() {
                // North pole: theta = 0 => cosTheta = 1, sinTheta = 0
                let north = sphericalDirection(sinTheta: 0, cosTheta: 1, phi: 0)
                #expect(abs(north.x) <= 1e-6)
                #expect(abs(north.y) <= 1e-6)
                #expect(abs(north.z - 1.0) <= 1e-6)
        }

        @Test func sphericalDirectionAtEquator() {
                // Equator at phi=0: sinTheta = 1, cosTheta = 0
                let equator = sphericalDirection(sinTheta: 1, cosTheta: 0, phi: 0)
                #expect(abs(equator.x - 1.0) <= 1e-6)
                #expect(abs(equator.y) <= 1e-6)
                #expect(abs(equator.z) <= 1e-6)
        }

        @Test func sphericalCoordinatesRoundTrip() {
                let original = normalized(Vector(x: 1, y: 2, z: 3))
                let (theta, phi) = sphericalCoordinatesFrom(vector: original)
                let reconstructed = sphericalDirection(
                        sinTheta: Glibc.sinf(theta), cosTheta: Glibc.cosf(theta), phi: phi)
                #expect(abs(original.x - reconstructed.x) <= 1e-5)
                #expect(abs(original.y - reconstructed.y) <= 1e-5)
                #expect(abs(original.z - reconstructed.z) <= 1e-5)
        }

        // MARK: - Max Dimension

        @Test func maxDimensionX() {
                let v = Vector3(x: 10, y: 2, z: 3)
                #expect(maxDimension(v) == 0)
        }

        @Test func maxDimensionY() {
                let v = Vector3(x: 1, y: 20, z: 3)
                #expect(maxDimension(v) == 1)
        }

        @Test func maxDimensionZ() {
                let v = Vector3(x: 1, y: 2, z: 30)
                #expect(maxDimension(v) == 2)
        }

        // MARK: - Union

        @Test func unionBoundPoint() {
                let bound = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                let point = Point(x: 2, y: -1, z: 0.5)
                let result = union(bound: bound, point: point)
                #expect(abs(result.pMin.x) <= 1e-6)
                #expect(abs(result.pMin.y - (-1)) <= 1e-6)
                #expect(abs(result.pMax.x - 2) <= 1e-6)
                #expect(abs(result.pMax.z - 1) <= 1e-6)
        }
}
