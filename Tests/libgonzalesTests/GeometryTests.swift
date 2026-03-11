import XCTest

@testable import libgonzales

final class GeometryTests: XCTestCase {

        // MARK: - Dot Product

        func testDotVectorNormalOrthogonal() {
                let vector = Vector(x: 1, y: 0, z: 0)
                let normal = Normal(x: 0, y: 1, z: 0)
                XCTAssertEqual(dot(vector, normal), 0, accuracy: 1e-6)
        }

        func testDotVectorNormalParallel() {
                let vector = Vector(x: 0, y: 0, z: 3)
                let normal = Normal(x: 0, y: 0, z: 2)
                XCTAssertEqual(dot(vector, normal), 6.0, accuracy: 1e-6)
        }

        func testDotNormalVectorCommutes() {
                let vector = Vector(x: 1, y: 2, z: 3)
                let normal = Normal(x: 4, y: 5, z: 6)
                XCTAssertEqual(dot(vector, normal), dot(normal, vector), accuracy: 1e-6)
        }

        // MARK: - AbsDot

        func testAbsDotAlwaysNonNegative() {
                let vector = Vector(x: 1, y: 0, z: 0)
                let normal = Normal(x: -1, y: 0, z: 0)
                let result = absDot(vector, normal)
                XCTAssertGreaterThanOrEqual(result, 0)
                XCTAssertEqual(result, 1.0, accuracy: 1e-6)
        }

        // MARK: - Faceforward

        func testFaceforwardNormalFlips() {
                let normal = Normal(x: 0, y: 0, z: -1)
                let reference = Vector(x: 0, y: 0, z: 1)
                let result = faceforward(normal: normal, comparedTo: reference)
                // dot(normal, reference) < 0, so should flip
                XCTAssertEqual(result.z, 1.0, accuracy: 1e-6)
        }

        func testFaceforwardNormalNoFlip() {
                let normal = Normal(x: 0, y: 0, z: 1)
                let reference = Vector(x: 0, y: 0, z: 1)
                let result = faceforward(normal: normal, comparedTo: reference)
                XCTAssertEqual(result.z, 1.0, accuracy: 1e-6)
        }

        func testFaceforwardVectorFlips() {
                let vector = Vector(x: 0, y: 0, z: -1)
                let normal = Normal(x: 0, y: 0, z: 1)
                let result = faceforward(vector: vector, comparedTo: normal)
                XCTAssertEqual(result.z, 1.0, accuracy: 1e-6)
        }

        // MARK: - MakeCoordinateSystem

        func testMakeCoordinateSystemOrthonormal() {
                let v1 = normalized(Vector(x: 1, y: 1, z: 0))
                let (v2, v3) = makeCoordinateSystem(from: v1)

                // Orthogonality
                XCTAssertEqual(dot(v1, v2), 0, accuracy: 1e-5)
                XCTAssertEqual(dot(v1, v3), 0, accuracy: 1e-5)
                XCTAssertEqual(dot(v2, v3), 0, accuracy: 1e-5)

                // Unit length
                XCTAssertEqual(length(v2), 1.0, accuracy: 1e-5)
                XCTAssertEqual(length(v3), 1.0, accuracy: 1e-5)
        }

        func testMakeCoordinateSystemFromZAxis() {
                let v1 = Vector(x: 0, y: 0, z: 1)
                let (v2, v3) = makeCoordinateSystem(from: v1)

                XCTAssertEqual(dot(v1, v2), 0, accuracy: 1e-5)
                XCTAssertEqual(dot(v1, v3), 0, accuracy: 1e-5)
                XCTAssertEqual(dot(v2, v3), 0, accuracy: 1e-5)
        }

        // MARK: - Spherical Coordinates

        func testSphericalDirectionAtPoles() {
                // North pole: theta = 0 => cosTheta = 1, sinTheta = 0
                let north = sphericalDirection(sinTheta: 0, cosTheta: 1, phi: 0)
                XCTAssertEqual(north.x, 0, accuracy: 1e-6)
                XCTAssertEqual(north.y, 0, accuracy: 1e-6)
                XCTAssertEqual(north.z, 1.0, accuracy: 1e-6)
        }

        func testSphericalDirectionAtEquator() {
                // Equator at phi=0: sinTheta = 1, cosTheta = 0
                let equator = sphericalDirection(sinTheta: 1, cosTheta: 0, phi: 0)
                XCTAssertEqual(equator.x, 1.0, accuracy: 1e-6)
                XCTAssertEqual(equator.y, 0, accuracy: 1e-6)
                XCTAssertEqual(equator.z, 0, accuracy: 1e-6)
        }

        func testSphericalCoordinatesRoundTrip() {
                let original = normalized(Vector(x: 1, y: 2, z: 3))
                let (theta, phi) = sphericalCoordinatesFrom(vector: original)
                let reconstructed = sphericalDirection(
                        sinTheta: sin(theta), cosTheta: cos(theta), phi: phi)
                XCTAssertEqual(original.x, reconstructed.x, accuracy: 1e-5)
                XCTAssertEqual(original.y, reconstructed.y, accuracy: 1e-5)
                XCTAssertEqual(original.z, reconstructed.z, accuracy: 1e-5)
        }

        // MARK: - Max Dimension

        func testMaxDimensionX() {
                let v = Vector3(x: 10, y: 2, z: 3)
                XCTAssertEqual(maxDimension(v), 0)
        }

        func testMaxDimensionY() {
                let v = Vector3(x: 1, y: 20, z: 3)
                XCTAssertEqual(maxDimension(v), 1)
        }

        func testMaxDimensionZ() {
                let v = Vector3(x: 1, y: 2, z: 30)
                XCTAssertEqual(maxDimension(v), 2)
        }

        // MARK: - Union

        func testUnionBoundPoint() {
                let bound = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                let point = Point(x: 2, y: -1, z: 0.5)
                let result = union(bound: bound, point: point)
                XCTAssertEqual(result.pMin.x, 0, accuracy: 1e-6)
                XCTAssertEqual(result.pMin.y, -1, accuracy: 1e-6)
                XCTAssertEqual(result.pMax.x, 2, accuracy: 1e-6)
                XCTAssertEqual(result.pMax.z, 1, accuracy: 1e-6)
        }
}
