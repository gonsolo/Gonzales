struct Hair {

        private func absorptionFrom(
                eumelaninConcentration: Real,
                pheomelaninConcentration: Real = 0
        ) -> RgbSpectrum {
                let eumelaninAbsorptionCoefficient = RgbSpectrum(red: 0.419, green: 0.697, blue: 1.37)
                let pheomelaninAbsorptionCoefficient = RgbSpectrum(red: 0.187, green: 0.4, blue: 1.05)
                let eumelaninAbsorption = eumelaninConcentration * eumelaninAbsorptionCoefficient
                let pheomelaninAbsorption = pheomelaninConcentration * pheomelaninAbsorptionCoefficient
                return eumelaninAbsorption + pheomelaninAbsorption
        }

        func getBsdf(interaction: SurfaceInteraction, arena: TextureArena) -> HairBsdf {
                let eumelanin = self.eumelanin.evaluateFloat(at: interaction, arena: arena)
                let absorption = absorptionFrom(eumelaninConcentration: eumelanin)
                let hOffsetValue = interaction.uvCoordinates[1]
                let alpha: Real = 2
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let hairBsdf = HairBsdf(
                        alpha: alpha,
                        hOffset: hOffsetValue,
                        absorption: absorption,
                        bsdfFrame: bsdfFrame)
                return hairBsdf
        }

        var eumelanin: Texture
}

extension Hair {
        static func create(
                parameters: ParameterDictionary, textures: [String: Texture], arena: inout TextureArena
        ) throws -> Hair {
                let eumelanin = try parameters.findRealTexture(
                        name: "eumelanin", textures: textures, arena: &arena, else: 1.3)
                return Hair(eumelanin: eumelanin)
        }
}
