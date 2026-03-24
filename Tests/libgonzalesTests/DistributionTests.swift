import Testing

@testable import libgonzales

@Suite struct Distribution1DTests {

        // MARK: - CDF Properties

        @Test func uniformDistributionCdfMonotonic() {
                let dist = PiecewiseConstant1D(values: [1, 2, 3, 4, 5])
                for i in 1..<dist.cdf.count {
                        #expect(
                                dist.cdf[i] >= dist.cdf[i - 1],
                                "CDF not monotonic at index \(i)")
                }
        }

        @Test func uniformDistributionCdfBounds() {
                let dist = PiecewiseConstant1D(values: [1, 2, 3, 4, 5])
                #expect(abs(dist.cdf[0]) <= 1e-6, "CDF should start at 0")
                #expect(abs(dist.cdf[dist.cdf.count - 1] - 1.0) <= 1e-6, "CDF should end at 1")
        }

        @Test func uniformDistributionIntegral() {
                let values: [Real] = [2, 4, 6, 8]
                let dist = PiecewiseConstant1D(values: values)
                // Integral = sum(values) / count = 20/4 = 5
                let expected: Real = values.reduce(0, +) / Real(values.count)
                #expect(abs(dist.integral - expected) <= 1e-5)
        }

        // MARK: - Continuous Sampling

        @Test func sampleContinuousInRange() {
                let dist = PiecewiseConstant1D(values: [1, 2, 3, 4, 5])
                let samples: [Real] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]
                for sample in samples {
                        let (value, pdf, _) = dist.sampleContinuous(sample: sample)
                        #expect(value >= 0, "Value below 0 for sample \(sample)")
                        #expect(value <= 1, "Value above 1 for sample \(sample)")
                        #expect(pdf > 0, "PDF should be positive for sample \(sample)")
                }
        }

        @Test func highValueBinSelectedMoreOften() {
                let dist = PiecewiseConstant1D(values: [1, 1, 100, 1, 1])
                // The high-value bin (index 2) should be hit by most samples
                var highBinCount = 0
                let totalSamples = 1000
                for i in 0..<totalSamples {
                        let sample = Real(i) / Real(totalSamples)
                        let (_, _, offset) = dist.sampleContinuous(sample: sample)
                        if offset == 2 { highBinCount += 1 }
                }
                #expect(
                        highBinCount > totalSamples / 2,
                        "High-value bin should be selected more than half the time, got \(highBinCount)")
        }

        // MARK: - Discrete Sampling

        @Test func sampleDiscreteValidOffset() {
                let values: [Real] = [1, 2, 3, 4, 5]
                let dist = PiecewiseConstant1D(values: values)
                let samples: [Real] = [0.0, 0.1, 0.5, 0.9, 0.99]
                for sample in samples {
                        let (offset, _, _) = dist.sampleDiscrete(sample: sample)
                        #expect(offset >= 0, "Offset below 0 for sample \(sample)")
                        #expect(
                                offset < values.count,
                                "Offset \(offset) exceeds count for sample \(sample)")
                }
        }

        @Test func sampleDiscreteRemappedInRange() {
                let dist = PiecewiseConstant1D(values: [1, 2, 3])
                let samples: [Real] = [0.1, 0.3, 0.5, 0.7, 0.9]
                for sample in samples {
                        let (_, _, uRemapped) = dist.sampleDiscrete(sample: sample)
                        #expect(uRemapped >= 0, "Remapped u below 0")
                        #expect(uRemapped <= 1, "Remapped u above 1")
                }
        }

        @Test func discretePdfSumsToOne() {
                let values: [Real] = [3, 1, 4, 1, 5]
                let dist = PiecewiseConstant1D(values: values)
                var pdfSum: Real = 0
                for i in 0..<values.count {
                        pdfSum += dist.discretePdf(offset: i)
                }
                #expect(abs(pdfSum - 1.0) <= 1e-5, "Discrete PDF sum \(pdfSum) != 1.0")
        }

        // MARK: - Edge Cases

        @Test func allZeroDistributionBecomesUniform() {
                let dist = PiecewiseConstant1D(values: [0, 0, 0, 0])
                // When integral == 0, CDF becomes uniform: cdf[i] = i/n
                let expected: Real = 1.0 / 4.0
                #expect(abs(dist.cdf[1] - expected) <= 1e-6)
                #expect(abs(dist.cdf[4] - 1.0) <= 1e-6)
        }

        @Test func singleValueDistribution() {
                let dist = PiecewiseConstant1D(values: [42])
                let (value, pdf, offset) = dist.sampleContinuous(sample: 0.5)
                #expect(value >= 0)
                #expect(value <= 1)
                #expect(pdf > 0)
                #expect(offset == 0)
        }
}

@Suite struct Distribution2DTests {

        @Test func distribution2dSampleInRange() {
                let data: [Real] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                let dist = PiecewiseConstant2D(data: data, width: 3, height: 3)
                let samples: [Point2f] = [
                        Point2f(x: 0.1, y: 0.1),
                        Point2f(x: 0.5, y: 0.5),
                        Point2f(x: 0.9, y: 0.9),
                ]
                for sample in samples {
                        let (uv, pdf) = dist.sampleContinuous(sample: sample)
                        #expect(uv.x >= 0 && uv.x <= 1, "U out of range")
                        #expect(uv.y >= 0 && uv.y <= 1, "V out of range")
                        #expect(pdf > 0, "PDF should be positive")
                }
        }

        @Test func distribution2dPdfNonNegative() {
                let data: [Real] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                let dist = PiecewiseConstant2D(data: data, width: 3, height: 3)
                let texCoords: [Point2f] = [
                        Point2f(x: 0.0, y: 0.0),
                        Point2f(x: 0.5, y: 0.5),
                        Point2f(x: 0.99, y: 0.99),
                ]
                for tc in texCoords {
                        let pdf = dist.pdf(texCoord: tc)
                        #expect(pdf >= 0, "PDF should be non-negative at \(tc)")
                }
        }

        @Test func distribution2dRespondsToWeights() {
                // Row 1 (index 1) has much higher values
                let data: [Real] = [
                        1, 1, 1,
                        100, 100, 100,
                        1, 1, 1,
                ]
                let dist = PiecewiseConstant2D(data: data, width: 3, height: 3)
                var middleRowCount = 0
                let totalSamples = 1000
                for i in 0..<totalSamples {
                        let sample = Point2f(
                                x: Real(i % 10) / 10.0,
                                y: Real(i) / Real(totalSamples))
                        let (uv, _) = dist.sampleContinuous(sample: sample)
                        // Middle row is y in [0.33, 0.67)
                        if uv.y >= 0.2 && uv.y <= 0.8 {
                                middleRowCount += 1
                        }
                }
                #expect(
                        middleRowCount > totalSamples / 3,
                        "Middle row should be sampled more often, got \(middleRowCount)")
        }
}
