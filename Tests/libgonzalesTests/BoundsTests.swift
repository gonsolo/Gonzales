import Testing

@testable import libgonzales

@Suite struct BoundsTests {

        // MARK: - Construction

        @Test func constructionOrdersMinMax() {
                let bounds = Bounds3f(
                        first: Point(x: 5, y: 5, z: 5),
                        second: Point(x: -1, y: -1, z: -1))
                #expect(abs(bounds.pMin.x - (-1)) <= 1e-6)
                #expect(abs(bounds.pMax.x - 5) <= 1e-6)
        }

        @Test func defaultConstructionIsInvalid() {
                let bounds = Bounds3f()
                // Default bounds has pMin = +inf, pMax = -inf (empty/invalid)
                #expect(bounds.pMin.x == FloatX.infinity)
                #expect(bounds.pMax.x == -FloatX.infinity)
        }

        // MARK: - Surface Area

        @Test func surfaceAreaUnitCube() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                #expect(abs(bounds.surfaceArea() - 6.0) <= 1e-6)
        }

        @Test func surfaceAreaRectangularBox() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 2, y: 3, z: 4))
                // 2*(2*3 + 2*4 + 3*4) = 2*(6 + 8 + 12) = 52
                #expect(abs(bounds.surfaceArea() - 52.0) <= 1e-6)
        }

        // MARK: - Diagonal and Extent

        @Test func diagonal() {
                let bounds = Bounds3f(
                        first: Point(x: 1, y: 2, z: 3),
                        second: Point(x: 4, y: 7, z: 13))
                let diag = bounds.diagonal()
                #expect(abs(diag.x - 3) <= 1e-6)
                #expect(abs(diag.y - 5) <= 1e-6)
                #expect(abs(diag.z - 10) <= 1e-6)
        }

        @Test func maximumExtentX() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 10, y: 1, z: 1))
                #expect(bounds.maximumExtent() == 0)
        }

        @Test func maximumExtentY() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 10, z: 1))
                #expect(bounds.maximumExtent() == 1)
        }

        @Test func maximumExtentZ() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 10))
                #expect(bounds.maximumExtent() == 2)
        }

        // MARK: - Center

        @Test func center() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 2, y: 4, z: 6))
                let c = bounds.center
                #expect(abs(c.x - 1.0) <= 1e-6)
                #expect(abs(c.y - 2.0) <= 1e-6)
                #expect(abs(c.z - 3.0) <= 1e-6)
        }

        // MARK: - Union

        @Test func unionOfTwoBounds() {
                let a = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                let b = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 0.5, y: 0.5, z: 0.5))
                let result = union(first: a, second: b)
                #expect(abs(result.pMin.x - (-1)) <= 1e-6)
                #expect(abs(result.pMin.y - (-1)) <= 1e-6)
                #expect(abs(result.pMin.z - (-1)) <= 1e-6)
                #expect(abs(result.pMax.x - 1) <= 1e-6)
                #expect(abs(result.pMax.y - 1) <= 1e-6)
                #expect(abs(result.pMax.z - 1) <= 1e-6)
        }

        // MARK: - Add Point

        @Test func addPointExpandsBounds() {
                var bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                bounds.add(point: Point(x: 3, y: -2, z: 0.5))
                #expect(abs(bounds.pMin.y - (-2)) <= 1e-6)
                #expect(abs(bounds.pMax.x - 3) <= 1e-6)
                // z should stay the same
                #expect(abs(bounds.pMin.z - 0) <= 1e-6)
                #expect(abs(bounds.pMax.z - 1) <= 1e-6)
        }

        // MARK: - Ray Intersection

        @Test func rayHitsUnitBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 5),
                        direction: Vector(x: 0, y: 0, z: -1))
                #expect(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        @Test func rayMissesBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                let ray = Ray(
                        origin: Point(x: 0, y: 5, z: 0),
                        direction: normalized(Vector(x: 0.01, y: 1, z: 0.01)))
                #expect(!bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        @Test func rayBehindBoxMisses() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 5),
                        direction: normalized(Vector(x: 0.001, y: 0.001, z: 1)))
                #expect(!bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        @Test func rayHitsWithTHitLimit() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 5),
                        direction: normalized(Vector(x: 0.001, y: 0.001, z: -1)))
                #expect(!bounds.intersects(ray: ray, tHit: 3.0))
                #expect(bounds.intersects(ray: ray, tHit: 10.0))
        }

        @Test func rayInsideBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 0),
                        direction: Vector(x: 1, y: 0, z: 0))
                #expect(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        @Test func diagonalRayHitsBox() {
                let bounds = Bounds3f(
                        first: Point(x: -1, y: -1, z: -1),
                        second: Point(x: 1, y: 1, z: 1))
                let direction = normalized(Vector(x: 1, y: 1, z: 1))
                let ray = Ray(
                        origin: Point(x: -5, y: -5, z: -5),
                        direction: direction)
                #expect(bounds.intersects(ray: ray, tHit: FloatX.infinity))
        }

        // MARK: - Subscript

        @Test func subscriptAccess() {
                let bounds = Bounds3f(
                        first: Point(x: 1, y: 2, z: 3),
                        second: Point(x: 4, y: 5, z: 6))
                #expect(abs(bounds[0].x - 1) <= 1e-6)  // pMin
                #expect(abs(bounds[1].x - 4) <= 1e-6)  // pMax
        }

        // MARK: - Expand

        @Test func expandBounds() {
                let bounds = Bounds3f(
                        first: Point(x: 0, y: 0, z: 0),
                        second: Point(x: 1, y: 1, z: 1))
                let expanded = libgonzales.expand(bounds: bounds, by: 0.5)
                #expect(abs(expanded.pMin.x - (-0.5)) <= 1e-6)
                #expect(abs(expanded.pMax.x - 1.5) <= 1e-6)
        }
}
