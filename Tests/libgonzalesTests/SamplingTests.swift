import XCTest

@testable import libgonzales

final class SamplingTests: XCTestCase {

        // MARK: - Power Heuristic

        func testPowerHeuristicEqualPdfs() {
                let result = powerHeuristic(pdfF: 1.0, pdfG: 1.0)
                XCTAssertEqual(result, 0.5, accuracy: 1e-6)
        }

        func testPowerHeuristicZeroPdfF() {
                let result = powerHeuristic(pdfF: 0.0, pdfG: 1.0)
                XCTAssertEqual(result, 0.0, accuracy: 1e-6)
        }

        func testPowerHeuristicZeroPdfG() {
                let result = powerHeuristic(pdfF: 1.0, pdfG: 0.0)
                XCTAssertEqual(result, 0.0, accuracy: 1e-6)
        }

        func testPowerHeuristicBothZero() {
                let result = powerHeuristic(pdfF: 0.0, pdfG: 0.0)
                XCTAssertEqual(result, 0.0, accuracy: 1e-6)
        }

        func testPowerHeuristicDominating() {
                // When pdfF >> pdfG, result should be close to 1
                let result = powerHeuristic(pdfF: 100.0, pdfG: 1.0)
                XCTAssertGreaterThan(result, 0.99)
        }

        func testPowerHeuristicDominated() {
                // When pdfF << pdfG, result should be close to 0
                let result = powerHeuristic(pdfF: 1.0, pdfG: 100.0)
                XCTAssertLessThan(result, 0.01)
        }

        func testPowerHeuristicKnownValue() {
                // pdfF=2, pdfG=3: result = 4 / (4+9) = 4/13
                let result = powerHeuristic(pdfF: 2.0, pdfG: 3.0)
                XCTAssertEqual(result, 4.0 / 13.0, accuracy: 1e-6)
        }

        // MARK: - Concentric Disk Sampling

        func testConcentricSampleDiskCenter() {
                // uSample = (0.5, 0.5) should map to origin
                let result = concentricSampleDisk(uSample: (0.5, 0.5))
                XCTAssertEqual(result.x, 0, accuracy: 1e-6)
                XCTAssertEqual(result.y, 0, accuracy: 1e-6)
        }

        func testConcentricSampleDiskWithinUnitCircle() {
                let samples: [TwoRandomVariables] = [
                        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                        (0.25, 0.75), (0.8, 0.2),
                ]
                for uSample in samples {
                        let result = concentricSampleDisk(uSample: uSample)
                        let radiusSquared = result.x * result.x + result.y * result.y
                        XCTAssertLessThanOrEqual(
                                radiusSquared, 1.0 + 1e-6,
                                "Sample \(uSample) mapped outside unit circle: r²=\(radiusSquared)")
                }
        }

        // MARK: - Cosine Hemisphere Sampling

        func testCosineSampleHemisphereUnitLength() {
                let result = cosineSampleHemisphere(uSample: (0.3, 0.7))
                let len = (result.x * result.x + result.y * result.y + result.z * result.z).squareRoot()
                XCTAssertEqual(len, 1.0, accuracy: 1e-5)
        }

        func testCosineSampleHemispherePositiveZ() {
                let samples: [TwoRandomVariables] = [
                        (0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.0, 0.0),
                ]
                for uSample in samples {
                        let result = cosineSampleHemisphere(uSample: uSample)
                        XCTAssertGreaterThanOrEqual(
                                result.z, 0,
                                "Sample \(uSample) produced negative z: \(result.z)")
                }
        }

        // MARK: - Random Sampler

        func testRandomSamplerValuesInRange() {
                var sampler = RandomSampler(numberOfSamples: 16)
                for _ in 0..<100 {
                        let value = sampler.get1D()
                        XCTAssertGreaterThanOrEqual(value, 0)
                        XCTAssertLessThan(value, 1)
                }
        }

        func testRandomSampler2DInRange() {
                var sampler = RandomSampler(numberOfSamples: 16)
                for _ in 0..<100 {
                        let (u, v) = sampler.get2D()
                        XCTAssertGreaterThanOrEqual(u, 0)
                        XCTAssertLessThan(u, 1)
                        XCTAssertGreaterThanOrEqual(v, 0)
                        XCTAssertLessThan(v, 1)
                }
        }

        func testRandomSamplerCloneIsIndependent() {
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
                XCTAssertFalse(allSame, "Cloned samplers should produce different sequences")
        }

        func testSamplerSamplesPerPixel() {
                let sampler = Sampler.random(RandomSampler(numberOfSamples: 64))
                XCTAssertEqual(sampler.samplesPerPixel, 64)
        }
}
