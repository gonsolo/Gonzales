import Foundation

final class MixBsdf: FramedBsdf, @unchecked Sendable {

        let bsdf1: BsdfVariant
        let bsdf2: BsdfVariant
        let amount: Real
        let bsdfFrame: BsdfFrame

        init(
                bsdf1: BsdfVariant,
                bsdf2: BsdfVariant,
                amount: Real,
                bsdfFrame: BsdfFrame
        ) {
                self.bsdf1 = bsdf1
                self.bsdf2 = bsdf2
                self.amount = clamp(value: amount, low: 0.0, high: 1.0)
                self.bsdfFrame = bsdfFrame
        }
}

extension MixBsdf {

        func evaluate(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                let eval1 = bsdf1.evaluate(outgoing: outgoing, incident: incident)
                let eval2 = bsdf2.evaluate(outgoing: outgoing, incident: incident)
                let weight1 = 1.0 - amount
                let weight2 = amount
                return RgbSpectrum(intensity: weight1) * eval1 + RgbSpectrum(intensity: weight2) * eval2
        }

        func probabilityDensity(outgoing: Vector, incident: Vector) -> Real {
                let pdf1 = bsdf1.probabilityDensity(outgoing: outgoing, incident: incident)
                let pdf2 = bsdf2.probabilityDensity(outgoing: outgoing, incident: incident)
                let weight1 = 1.0 - amount
                let weight2 = amount
                return weight1 * pdf1 + weight2 * pdf2
        }

        func sample(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                // Sample according to weights
                let weight1 = 1.0 - amount

                // We use one dimension of uSample to choose which BSDF to sample
                let selectionVar = uSample.0
                let remappedU0: Real

                let bsdfSample: BsdfSample
                
                if selectionVar < weight1 {
                        remappedU0 = min(selectionVar / weight1, 1.0 - .ulpOfOne)
                        let newUSample = (remappedU0, uSample.1, uSample.2)
                        bsdfSample = bsdf1.sample(outgoing: outgoing, uSample: newUSample)
                } else {
                        let weight2 = amount
                        if weight2 == 0 { return BsdfSample() } // Safe fallback
                        remappedU0 = min((selectionVar - weight1) / weight2, 1.0 - .ulpOfOne)
                        let newUSample = (remappedU0, uSample.1, uSample.2)
                        bsdfSample = bsdf2.sample(outgoing: outgoing, uSample: newUSample)
                }

                // But the pdf and estimate must reflect the mixture!
                guard !bsdfSample.estimate.isBlack && bsdfSample.probabilityDensity > 0 else {
                        return BsdfSample()
                }

                let incident = bsdfSample.incoming
                if  bsdf1.isSpecular && bsdf2.isSpecular {
                        return bsdfSample // Delta mixtures handled carefully
                }
                
                let eval = evaluate(outgoing: outgoing, incident: incident)
                let pdf = probabilityDensity(outgoing: outgoing, incident: incident)
                if pdf == 0 { return BsdfSample() }

                return BsdfSample(eval, incident, pdf)
        }

        func albedo() -> RgbSpectrum {
                return RgbSpectrum(intensity: 1.0 - amount) * bsdf1.albedo() + RgbSpectrum(intensity: amount) * bsdf2.albedo()
        }

        var isSpecular: Bool {
                // If both are specular, the mixture is effectively specular.
                // Otherwise it's a mixture.
                return bsdf1.isSpecular && bsdf2.isSpecular
        }
}
