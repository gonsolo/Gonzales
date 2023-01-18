final class Mirror: Material {

        init(kr: RGBSpectrumTexture) { self.kr = kr }

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let reflectance = kr.evaluateRGBSpectrum(at: interaction)
                bsdf.set(bxdf: SpecularReflection(reflectance: reflectance, fresnel: FresnelNop()))
                return bsdf
        }

        var kr: RGBSpectrumTexture
}

func createMirror(parameters: ParameterDictionary) throws -> Mirror {
        let kr = try parameters.findRGBSpectrumTexture(name: "Kr")
        return Mirror(kr: kr)
}
