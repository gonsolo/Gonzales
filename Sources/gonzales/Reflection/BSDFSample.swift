struct BSDFSample {

        init(
                _ estimate: RGBSpectrum = black,
                _ incoming: Vector = nullVector,
                _ probabilityDensity: FloatX = 0
        ) {
                self.estimate = estimate
                self.incoming = incoming
                self.probabilityDensity = probabilityDensity
        }

        func throughputWeight(normal: Normal = upNormal) -> RGBSpectrum {
                estimate * absDot(incoming, normal) / probabilityDensity
        }

        var isValid: Bool {
                !estimate.isBlack && incoming.z != 0 && probabilityDensity != 0
        }

        func isReflection(wo: Vector) -> Bool {
                sameHemisphere(incoming, wo)
        }

        var estimate: RGBSpectrum
        let incoming: Vector
        var probabilityDensity: FloatX
}

let invalidBSDFSample = BSDFSample()
