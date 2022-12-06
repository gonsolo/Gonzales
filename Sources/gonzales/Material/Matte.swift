final class Matte: Material {

        init(kd: SpectrumTexture) { self.kd = kd }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let kde = kd.evaluateSpectrum(at: interaction)
                bsdf.set(bxdf: LambertianReflection(reflectance: kde))
                return (bsdf, nil)
        }

        var kd: SpectrumTexture
}

func createMatte(parameters: ParameterDictionary) throws -> Matte {
        if let reflectance = try parameters.findSpectrum(name: "reflectance", else: nil) {
                let kd = ConstantTexture<Spectrum>(value: reflectance)
                return Matte(kd: kd)
        }
        let kd: SpectrumTexture = try parameters.findSpectrumTexture(name: "Kd", else: gray)
        return Matte(kd: kd)
}
