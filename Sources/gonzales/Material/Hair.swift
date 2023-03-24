final class Hair: Material {

        init(eumelanin: FloatTexture) {
                self.eumelanin = eumelanin
        }

        private func absorptionFrom(
                eumelaninConcentration: FloatX,
                pheomelaninConcentration: FloatX = 0
        ) -> RGBSpectrum {
                let eumelaninAbsorptionCoefficient = RGBSpectrum(r: 0.419, g: 0.697, b: 1.37)
                let pheomelaninAbsorptionCoefficient = RGBSpectrum(r: 0.187, g: 0.4, b: 1.05)
                let eumelaninAbsorption = eumelaninConcentration * eumelaninAbsorptionCoefficient
                let pheomelaninAbsorption = pheomelaninConcentration * pheomelaninAbsorptionCoefficient
                return eumelaninAbsorption + pheomelaninAbsorption
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                let eumelanin = self.eumelanin.evaluateFloat(at: interaction)
                let absorption = absorptionFrom(eumelaninConcentration: eumelanin)
                // Embree already provides values from -1 to 1 for flat bspline curves
                let h = interaction.uv[1]
                var bsdf = BSDF(interaction: interaction)
                let alpha: FloatX = 2
                bsdf.set(bxdf: HairBsdf(alpha: alpha, h: h, absorption: absorption))
                return bsdf
        }

        var eumelanin: FloatTexture
}

func createHair(parameters: ParameterDictionary) throws -> Hair {
        let eumelanin = try parameters.findFloatXTexture(name: "eumelanin", else: 1.3)
        return Hair(eumelanin: eumelanin)
}
