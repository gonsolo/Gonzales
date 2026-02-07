struct Hair {

        private func absorptionFrom(
                eumelaninConcentration: FloatX,
                pheomelaninConcentration: FloatX = 0
        ) -> RgbSpectrum {
                let eumelaninAbsorptionCoefficient = RgbSpectrum(red: 0.419, green: 0.697, blue: 1.37)
                let pheomelaninAbsorptionCoefficient = RgbSpectrum(red: 0.187, green: 0.4, blue: 1.05)
                let eumelaninAbsorption = eumelaninConcentration * eumelaninAbsorptionCoefficient
                let pheomelaninAbsorption = pheomelaninConcentration * pheomelaninAbsorptionCoefficient
                return eumelaninAbsorption + pheomelaninAbsorption
        }

        func getBsdf(interaction: any Interaction) -> HairBsdf {
                let eumelanin = self.eumelanin.evaluateFloat(at: interaction)
                let absorption = absorptionFrom(eumelaninConcentration: eumelanin)
                // Embree already provides values from -1 to 1 for flat bspline curves
                let hOffsetValue = interaction.uvCoordinates[1]
                let alpha: FloatX = 2
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let hairBsdf = HairBsdf(
                        alpha: alpha,
                        hOffset: hOffsetValue,
                        absorption: absorption,
                        bsdfFrame: bsdfFrame)
                return hairBsdf
        }

        var eumelanin: FloatTexture
}

@MainActor
func createHair(parameters: ParameterDictionary) throws -> Hair {
        let eumelanin = try parameters.findFloatXTexture(name: "eumelanin", else: 1.3)
        return Hair(eumelanin: eumelanin)
}
