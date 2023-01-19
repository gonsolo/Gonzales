final class Matte: Material {

        init(kd: RGBSpectrumTexture) { self.kd = kd }

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let kde = kd.evaluateRGBSpectrum(at: interaction)
                bsdf.set(bxdf: LambertianReflection(reflectance: kde))
                return bsdf
        }

        var kd: RGBSpectrumTexture
}

func createMatte(parameters: ParameterDictionary) throws -> Matte {
        if let reflectance = try parameters.findSpectrum(name: "reflectance", else: nil)
                as? RGBSpectrum
        {
                let kd = ConstantTexture<RGBSpectrum>(value: reflectance)
                return Matte(kd: kd)
        }
        let kd: RGBSpectrumTexture = try parameters.findRGBSpectrumTexture(name: "Kd", else: gray)
        return Matte(kd: kd)
}
