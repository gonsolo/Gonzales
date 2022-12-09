final class Mirror: Material {

        init(kr: SpectrumTexture) { self.kr = kr }

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let reflectance = kr.evaluateSpectrum(at: interaction)
                bsdf.set(bxdf: SpecularReflection(reflectance: reflectance, fresnel: FresnelNop()))
                return bsdf
        }

        var kr: SpectrumTexture
}

func createMirror(parameters: ParameterDictionary) throws -> Mirror {
        let kr = try parameters.findSpectrumTexture(name: "Kr")
        return Mirror(kr: kr)
}
