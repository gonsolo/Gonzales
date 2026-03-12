import Testing

@testable import libgonzales

@Suite struct FilterTests {

        // MARK: - BoxFilter

        @Test func boxFilterEvaluateAlwaysOne() {
                let filter = BoxFilter(support: Vector2F(x: 1, y: 1))
                #expect(abs(filter.evaluate(atLocation: Point2f(x: 0, y: 0)) - 1) <= 1e-6)
                #expect(abs(filter.evaluate(atLocation: Point2f(x: 0.5, y: 0.5)) - 1) <= 1e-6)
                #expect(abs(filter.evaluate(atLocation: Point2f(x: 99, y: 99)) - 1) <= 1e-6)
        }

        @Test func boxFilterSampleCenter() {
                let filter = BoxFilter(support: Vector2F(x: 2, y: 2))
                // uSample (0.5, 0.5) -> center -> (0, 0)
                let sample = filter.sample(uSample: (0.5, 0.5))
                #expect(abs(sample.location.x) <= 1e-6)
                #expect(abs(sample.location.y) <= 1e-6)
        }

        @Test func boxFilterSampleWithinSupport() {
                let support = Vector2F(x: 2, y: 3)
                let filter = BoxFilter(support: support)
                let samples: [(Real, Real)] = [
                        (0, 0), (1, 1), (0.5, 0.5), (0.25, 0.75),
                ]
                for uSample in samples {
                        let result = filter.sample(uSample: uSample)
                        #expect(abs(result.location.x) <= support.x + 1e-6)
                        #expect(abs(result.location.y) <= support.y + 1e-6)
                }
        }

        // MARK: - TriangleFilter

        @Test func triangleFilterEvaluateAtCenter() {
                let filter = TriangleFilter(support: Vector2F(x: 2, y: 2))
                // At center (0,0): max(0, 2-0) * max(0, 2-0) = 4
                let value = filter.evaluate(atLocation: Point2f(x: 0, y: 0))
                #expect(abs(value - 4) <= 1e-6)
        }

        @Test func triangleFilterEvaluateAtEdge() {
                let filter = TriangleFilter(support: Vector2F(x: 2, y: 2))
                // At edge (2,0): max(0, 2-2) * max(0, 2-0) = 0
                let value = filter.evaluate(atLocation: Point2f(x: 2, y: 0))
                #expect(abs(value) <= 1e-6)
        }

        @Test func triangleFilterEvaluateOutsideSupport() {
                let filter = TriangleFilter(support: Vector2F(x: 1, y: 1))
                // Outside support (3,3): max(0, 1-3) = 0
                let value = filter.evaluate(atLocation: Point2f(x: 3, y: 3))
                #expect(abs(value) <= 1e-6)
        }

        @Test func triangleFilterSymmetric() {
                let filter = TriangleFilter(support: Vector2F(x: 2, y: 2))
                let pos = filter.evaluate(atLocation: Point2f(x: 0.5, y: 0.7))
                let neg = filter.evaluate(atLocation: Point2f(x: -0.5, y: -0.7))
                #expect(abs(pos - neg) <= 1e-6)
        }

        // MARK: - GaussianFilter

        @Test func gaussianFilterCenterLargerThanEdge() {
                let filter = GaussianFilter(
                        withSupport: Vector2F(x: 2, y: 2), withSigma: 1)
                let center = filter.evaluate(atLocation: Point2f(x: 0, y: 0))
                let edge = filter.evaluate(atLocation: Point2f(x: 1.5, y: 1.5))
                #expect(center > edge)
        }

        @Test func gaussianFilterNonNegative() {
                let filter = GaussianFilter(
                        withSupport: Vector2F(x: 2, y: 2), withSigma: 1)
                let samples: [Point2f] = [
                        Point2f(x: 0, y: 0),
                        Point2f(x: 1, y: 0),
                        Point2f(x: 0, y: 1),
                        Point2f(x: 1.9, y: 1.9),
                ]
                for p in samples {
                        #expect(filter.evaluate(atLocation: p) >= 0)
                }
        }

        @Test func gaussianFilterSymmetric() {
                let filter = GaussianFilter(
                        withSupport: Vector2F(x: 2, y: 2), withSigma: 1)
                let pos = filter.evaluate(atLocation: Point2f(x: 0.5, y: 0.7))
                let neg = filter.evaluate(atLocation: Point2f(x: -0.5, y: -0.7))
                #expect(abs(pos - neg) <= 1e-6)
        }

        @Test func gaussianFilterSampleWithinSupport() {
                let support = Vector2F(x: 2, y: 2)
                let filter = GaussianFilter(withSupport: support, withSigma: 1)
                let samples: [(Real, Real)] = [
                        (0.1, 0.1), (0.5, 0.5), (0.9, 0.9),
                ]
                for uSample in samples {
                        let result = filter.sample(uSample: uSample)
                        #expect(abs(result.location.x) <= support.x + 1e-6)
                        #expect(abs(result.location.y) <= support.y + 1e-6)
                        #expect(result.probabilityDensity > 0)
                }
        }
}
