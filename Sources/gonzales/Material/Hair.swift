final class Hair: Material {

        init(eumelanin: FloatTexture) {
                self.eumelanin = eumelanin
        }

        private func sigmaAFromConcentration(eumelanin: FloatX, pheomelanin: FloatX = 0)
                -> RGBSpectrum
        {
                let eumelaninSigmaA = RGBSpectrum(r: 0.419, g: 0.697, b: 1.37)
                let pheomelaninSigmaA = RGBSpectrum(r: 0.187, g: 0.4, b: 1.05)
                let sigmaA = eumelanin * eumelaninSigmaA + pheomelanin * pheomelaninSigmaA
                return sigmaA
        }

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                let eumelanin = self.eumelanin.evaluateFloat(at: interaction)
                let sigmaA = sigmaAFromConcentration(eumelanin: eumelanin)
                let h = -1 + 2 * interaction.uv[1]
                print("Hair uv: ", interaction.uv[1])
                var bsdf = BSDF(interaction: interaction)
                let alpha: FloatX = 2
                bsdf.set(bxdf: HairBsdf(alpha: alpha, h: h, sigmaA: sigmaA))
                return bsdf
        }

        var eumelanin: FloatTexture
}

func createHair(parameters: ParameterDictionary) throws -> Hair {
        let eumelanin = try parameters.findFloatXTexture(name: "eumelanin", else: 1.3)
        return Hair(eumelanin: eumelanin)
}
