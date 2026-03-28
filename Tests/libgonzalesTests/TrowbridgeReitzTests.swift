import Testing

@testable import libgonzales

@Suite struct TrowbridgeReitzTests {

        private func makeDistribution(alpha: Real = 0.5) -> TrowbridgeReitzDistribution {
                return TrowbridgeReitzDistribution(alpha: (alpha, alpha))
        }

        // MARK: - Differential Area

        @Test func differentialAreaNonNegative() {
                let dist = makeDistribution()
                let halfVectors: [Vector] = [
                        Vector(x: 0, y: 0, z: 1),
                        normalized(Vector(x: 1, y: 0, z: 1)),
                        normalized(Vector(x: 0, y: 1, z: 1)),
                        normalized(Vector(x: 1, y: 1, z: 1)),
                ]
                for half in halfVectors {
                        #expect(
                                dist.differentialArea(withNormal: half) >= 0,
                                "D(wh) should be non-negative")
                }
        }

        @Test func differentialAreaPeaksAtNormal() {
                let dist = makeDistribution(alpha: 0.1)
                let peakValue = dist.differentialArea(withNormal: Vector(x: 0, y: 0, z: 1))
                let offAngle = dist.differentialArea(
                        withNormal: normalized(Vector(x: 0.5, y: 0, z: 1)))
                #expect(peakValue >= offAngle, "D(n) should be peak value")
        }

        // MARK: - isSmooth

        @Test func isSmoothForSmallAlpha() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.0005, 0.0005))
                #expect(dist.isSmooth)
        }

        @Test func isNotSmoothForLargeAlpha() {
                let dist = TrowbridgeReitzDistribution(alpha: (0.5, 0.5))
                #expect(!dist.isSmooth)
        }

        // MARK: - Sampling

        @Test func sampleHalfVectorIsNormalized() {
                let dist = makeDistribution()
                let outgoing = normalized(Vector(x: 0, y: 0, z: 1))
                let samples: [TwoRandomVariables] = [
                        (0.1, 0.2), (0.5, 0.5), (0.9, 0.1),
                ]
                for uSample in samples {
                        let half = dist.sampleHalfVector(outgoing: outgoing, uSample: uSample)
                        let len = length(half)
                        #expect(abs(len - 1.0) <= 1e-4, "Half-vector length \(len) != 1.0")
                }
        }

        // MARK: - Lambda

        @Test func lambdaNonNegative() {
                let dist = makeDistribution()
                let vectors: [Vector] = [
                        Vector(x: 0, y: 0, z: 1),
                        normalized(Vector(x: 1, y: 0, z: 1)),
                        normalized(Vector(x: 0, y: 1, z: 0.1)),
                ]
                for v in vectors {
                        #expect(dist.lambda(v) >= 0, "Lambda should be non-negative")
                }
        }

        // MARK: - Alpha conversion

        @Test func getAlphaFromRoughness() {
                let roughness: Real = 0.25
                let alpha = TrowbridgeReitzDistribution.getAlpha(from: roughness)
                #expect(abs(alpha - 0.5) <= 1e-5)  // sqrt(0.25) = 0.5
        }
}
