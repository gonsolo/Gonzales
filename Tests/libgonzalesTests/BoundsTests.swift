import XCTest

@testable import libgonzales

final class BoundsTests: XCTestCase {

        // MARK: - Construction

        func testConstructionOrdersMinMax() {
                let bounds = Bounds3f(
                        first: Point(x: 5, y: 5, z: 5),
                        second: Point(x: -1, y: -1, z: -1))
                XCTAssertEqual(bounds.pMin.x, -1, accuracy: 1e-6)
                XCTAssertEqual(bounds.pMax.x, 5, accuracy: 1e-6)
        }

        func testDefaultConstructionIsInvalid() {
                let bounds = Bounds3f()
                // Default bounds has pMin = +inf, pMax = -inf (empty/invalid)
                XCTAssertEqual(bounds.pMin.x, FloatX.infinity)
                XCTAssertEqual(bounds.pMax.x, -FloatX.infinity)
        }

        // MARK: - Surface Area

        func testSurfaceAreaUnitCube() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                XCTAssertEqual(bounds.surfaceArea(), 6.0, accuracy: 1e-6)
        }

        func testSurfaceAreaRectangularBox() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 2, y: 3, z: 4))
                // 2*(2*3 + 2*4 + 3*4) = 2*(6 + 8 + 12) = 52
                XCTAssertEqual(bounds.surfaceArea(), 52.0, accuracy: 1e-6)
        }

        // MARK: - Diagonal and Extent

        func testDiagonal() {
                let bounds = Bounds3f(
                        first: Point(x: 1, y: 2, z: 3),
                        second: Point(x: 4, y: 7, z: 13))
                let diag = bounds.diagonal()
                XCTAssertEqual(diag.x, 3, accuracy: 1e-6)
                XCTAssertEqual(diag.y, 5, accuracy: 1e-6)
                XCTAssertEqual(diag.z, 10, accuracy: 1e-6)
        }

        func testMaximumExtentX() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 10, y: 1, z: 1))
                XCTAssertEqual(bounds.maximumExtent(), 0)
        }

        func testMaximumExtentY() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 10, z: 1))
                XCTAssertEqual(bounds.maximumExtent(), 1)
        }

        func testMaximumExtentZ() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 10))
                XCTAssertEqual(bounds.maximumExtent(), 2)
        }

        // MARK: - Center

        func testCenter() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 2, y: 4, z: 6))
                let c = bounds.center
                XCTAssertEqual(c.x, 1.0, accuracy: 1e-6)
                XCTAssertEqual(c.y, 2.0, accuracy: 1e-6)
                XCTAssertEqual(c.z, 3.0, accuracy: 1e-6)
        }

        // MARK: - Union

        func testUnionOfTwoBounds() {
                let a = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                let b = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 0.5, y: 0.5, z: 0.5))
                let result = union(first: a, second: b)
                XCTAssertEqual(result.pMin.x, -1, accuracy: 1e-6)
                XCTAssertEqual(result.pMin.y, -1, accuracy: 1e-6)
                XCTAssertEqual(result.pMin.z, -1, accuracy: 1e-6)
                XCTAssertEqual(result.pMax.x, 1, accuracy: 1e-6)
                XCTAssertEqual(result.pMax.y, 1, accuracy: 1e-6)
                XCTAssertEqual(result.pMax.z, 1, accuracy: 1e-6)
        }

        // MARK: - Add Point

        func testAddPointExpandsBounds() {
                var bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                bounds.add(point: Point(x: 3, y: -2, z: 0.5))
                XCTAssertEqual(bounds.pMin.y, -2, accuracy: 1e-6)
                XCTAssertEqual(bounds.pMax.x, 3, accuracy: 1e-6)
                // z should stay the same
                XCTAssertEqual(bounds.pMin.z, 0, accuracy: 1e-6)
                XCTAssertEqual(bounds.pMax.z, 1, accuracy: 1e-6)
        }

        // MARK: - Ray Intersection

        func testRayHitsUnitBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                // Ray from z=5 pointing toward origin
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 5),
                        direction: Vector(x: 0, y: 0, z: -1))
                XCTAssertTrue(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        func testRayMissesBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                // Ray from above the box, heading further away (all non-zero direction)
                let ray = Ray(
                        origin: Point(x: 0, y: 5, z: 0),
                        direction: normalized(Vector(x: 0.01, y: 1, z: 0.01)))
                XCTAssertFalse(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        func testRayBehindBoxMisses() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                // Ray past the box in z, heading further away (all non-zero direction)
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 5),
                        direction: normalized(Vector(x: 0.001, y: 0.001, z: 1)))
                XCTAssertFalse(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        func testRayHitsWithTHitLimit() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                // Non-axis-aligned ray heading toward box from z=5
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 5),
                        direction: normalized(Vector(x: 0.001, y: 0.001, z: -1)))
                // tHit = 3 means we stop before reaching the box (box starts at z≈1, t≈4)
                XCTAssertFalse(bounds.intersects(ray: ray, tHit: 3.0))
                // tHit = 10 should allow the hit
                XCTAssertTrue(bounds.intersects(ray: ray, tHit: 10.0))
        }

        func testRayInsideBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                // Ray starting inside the box
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 0),
                        direction: Vector(x: 1, y: 0, z: 0))
                XCTAssertTrue(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        func testDiagonalRayHitsBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                let direction = normalized(Vector(x: 1, y: 1, z: 1))
                let ray = Ray(
                        origin: Point(x: -5, y: -5, z: -5),
                        direction: direction)
                XCTAssertTrue(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        // MARK: - Subscript

        func testSubscript() {
                let bounds = Bounds3f(
                        first: Point(x: 1, y: 2, z: 3),
                        second: Point(x: 4, y: 5, z: 6))
                XCTAssertEqual(bounds[0].x, 1, accuracy: 1e-6)  // pMin
                XCTAssertEqual(bounds[1].x, 4, accuracy: 1e-6)  // pMax
        }

        // MARK: - Expand

        func testExpand() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                let expanded = expand(bounds: bounds, by: 0.5)
                XCTAssertEqual(expanded.pMin.x, -0.5, accuracy: 1e-6)
                XCTAssertEqual(expanded.pMax.x, 1.5, accuracy: 1e-6)
        }
}
