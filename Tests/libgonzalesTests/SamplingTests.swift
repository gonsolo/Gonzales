import Testing

@testable import libgonzales

@Suite struct SamplingTests {

        // MARK: - Power Heuristic

        @Test func powerHeuristicEqualPdfs() {
                let result = powerHeuristic(pdfF: 1.0, pdfG: 1.0)
                #expect(abs(result - 0.5) <= 1e-6)
        }

        @Test func powerHeuristicZeroPdfF() {
                let result = powerHeuristic(pdfF: 0.0, pdfG: 1.0)
                #expect(abs(result) <= 1e-6)
        }

        @Test func powerHeuristicZeroPdfG() {
                let result = powerHeuristic(pdfF: 1.0, pdfG: 0.0)
                #expect(abs(result) <= 1e-6)
        }

        @Test func powerHeuristicBothZero() {
                let result = powerHeuristic(pdfF: 0.0, pdfG: 0.0)
                #expect(abs(result) <= 1e-6)
        }

        @Test func powerHeuristicDominating() {
                // When pdfF >> pdfG, result should be close to 1
                let result = powerHeuristic(pdfF: 100.0, pdfG: 1.0)
                #expect(result > 0.99)
        }

        @Test func powerHeuristicDominated() {
                // When pdfF << pdfG, result should be close to 0
                let result = powerHeuristic(pdfF: 1.0, pdfG: 100.0)
                #expect(result < 0.01)
        }

        @Test func powerHeuristicKnownValue() {
                // pdfF=2, pdfG=3: result = 4 / (4+9) = 4/13
                let result = powerHeuristic(pdfF: 2.0, pdfG: 3.0)
                #expect(abs(result - 4.0 / 13.0) <= 1e-6)
        }

        // MARK: - Concentric Disk Sampling

        @Test func concentricSampleDiskCenter() {
                // uSample = (0.5, 0.5) should map to origin
                let result = concentricSampleDisk(uSample: (0.5, 0.5))
                #expect(abs(result.x) <= 1e-6)
                #expect(abs(result.y) <= 1e-6)
        }

        @Test func concentricSampleDiskWithinUnitCircle() {
                let samples: [TwoRandomVariables] = [
                        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                        (0.25, 0.75), (0.8, 0.2),
                ]
                for uSample in samples {
                        let result = concentricSampleDisk(uSample: uSample)
                        let radiusSquared = result.x * result.x + result.y * result.y
                        #expect(
                                radiusSquared <= 1.0 + 1e-6,
                                "Sample \(uSample) mapped outside unit circle: r²=\(radiusSquared)")
                }
        }

        // MARK: - Cosine Hemisphere Sampling

        @Test func cosineSampleHemisphereUnitLength() {
                let result = cosineSampleHemisphere(uSample: (0.3, 0.7))
                let len = (result.x * result.x + result.y * result.y + result.z * result.z)
                        .squareRoot()
                #expect(abs(len - 1.0) <= 1e-5)
        }

        @Test func cosineSampleHemispherePositiveZ() {
                let samples: [TwoRandomVariables] = [
                        (0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.0, 0.0),
                ]
                for uSample in samples {
                        let result = cosineSampleHemisphere(uSample: uSample)
                        #expect(
                                result.z >= 0,
                                "Sample \(uSample) produced negative z: \(result.z)")
                }
        }

        // MARK: - Random Sampler

        @Test func randomSamplerValuesInRange() {
                var sampler = RandomSampler(numberOfSamples: 16)
                for _ in 0..<100 {
                        let value = sampler.get1D()
                        #expect(value >= 0)
                        #expect(value < 1)
                }
        }

        @Test func randomSampler2DInRange() {
                var sampler = RandomSampler(numberOfSamples: 16)
                for _ in 0..<100 {
                        let (u, v) = sampler.get2D()
                        #expect(u >= 0)
                        #expect(u < 1)
                        #expect(v >= 0)
                        #expect(v < 1)
                }
        }

        @Test func randomSamplerCloneIsIndependent() {
                let sampler = RandomSampler(numberOfSamples: 8)
                var clone1 = sampler.clone()
                var clone2 = sampler.clone()
                // Clones should produce different sequences (new RNG state)
                var allSame = true
                for _ in 0..<10 {
                        if clone1.get1D() != clone2.get1D() {
                                allSame = false
                                break
                        }
                }
                // It's astronomically unlikely that 10 random floats are all equal
                #expect(!allSame, "Cloned samplers should produce different sequences")
        }

        @Test func samplerSamplesPerPixel() {
                let sampler = Sampler.random(RandomSampler(numberOfSamples: 64))
                #expect(sampler.samplesPerPixel == 64)
        }
}
