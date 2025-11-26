public struct BsdfSample: Sendable {

        init(
                _ estimate: RgbSpectrum = black,
                _ incoming: Vector = nullVector,
                _ probabilityDensity: FloatX = 0
        ) {
                self.estimate = estimate
                self.incoming = incoming
                self.probabilityDensity = probabilityDensity
        }

        func throughputWeight(normal: Normal = upNormal) -> RgbSpectrum {
                estimate * absDot(incoming, normal) / probabilityDensity
        }

        var isValid: Bool {
                !estimate.isBlack && incoming.z != 0 && probabilityDensity != 0
        }

        func isReflection(wo: Vector) -> Bool {
                sameHemisphere(incoming, wo)
        }

        func isTransmission(wo: Vector) -> Bool {
                return !isReflection(wo: wo)
        }

        var estimate: RgbSpectrum
        let incoming: Vector
        var probabilityDensity: FloatX
}

let invalidBsdfSample = BsdfSample()
