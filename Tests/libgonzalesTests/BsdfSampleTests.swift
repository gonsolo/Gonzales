import Testing

@testable import libgonzales

@Suite struct BsdfSampleTests {

        @Test func invalidSampleIsNotValid() {
                let sample = BsdfSample()
                #expect(!sample.isValid)
        }

        @Test func validSampleIsValid() {
                let sample = BsdfSample(
                        RgbSpectrum(intensity: 0.5),
                        Vector(x: 0, y: 0, z: 1),
                        0.5)
                #expect(sample.isValid)
        }

        @Test func throughputWeightComputation() {
                let estimate = RgbSpectrum(intensity: 2.0)
                let incoming = Vector(x: 0, y: 0, z: 1)
                let pdf: Real = 0.5
                let sample = BsdfSample(estimate, incoming, pdf)
                let normal = Normal(x: 0, y: 0, z: 1)
                let weight = sample.throughputWeight(normal: normal)
                // estimate * absDot(incoming, normal) / pdf = 2 * 1 / 0.5 = 4
                #expect(abs(weight.red - 4.0) <= 1e-5)
                #expect(abs(weight.green - 4.0) <= 1e-5)
                #expect(abs(weight.blue - 4.0) <= 1e-5)
        }

        @Test func isReflectionWhenSameHemisphere() {
                let sample = BsdfSample(
                        white,
                        Vector(x: 0, y: 0, z: 1),
                        1.0)
                let outgoing = Vector(x: 0, y: 0, z: 0.5)
                #expect(sample.isReflection(outgoing: outgoing))
                #expect(!sample.isTransmission(outgoing: outgoing))
        }

        @Test func isTransmissionWhenDifferentHemisphere() {
                let sample = BsdfSample(
                        white,
                        Vector(x: 0, y: 0, z: 1),
                        1.0)
                let outgoing = Vector(x: 0, y: 0, z: -0.5)
                #expect(!sample.isReflection(outgoing: outgoing))
                #expect(sample.isTransmission(outgoing: outgoing))
        }
}
