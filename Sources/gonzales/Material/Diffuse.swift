enum DiffuseError: Error {
        case noReflectance
        case noTexture
}

final class Diffuse: Material {

        init(reflectance: RGBSpectrumTexture) {
                self.reflectance = reflectance
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let reflectance = reflectance.evaluateRGBSpectrum(at: interaction)
                bsdf.set(bxdf: DiffuseBsdf(reflectance: reflectance))
                return bsdf
        }

        let reflectance: RGBSpectrumTexture
}

func createDiffuse(parameters: ParameterDictionary) throws -> Diffuse {
        let reflectanceTextureName = try parameters.findTexture(name: "reflectance")
        if !reflectanceTextureName.isEmpty {
                guard let texture = state.textures[reflectanceTextureName] as? RGBSpectrumTexture else {
                        throw DiffuseError.noTexture
                }
                return Diffuse(reflectance: texture)
        }
        if let reflectanceSpectrum = try parameters.findSpectrum(name: "reflectance", else: nil)
                as? RGBSpectrum
        {
                let reflectanceConstant = ConstantTexture<RGBSpectrum>(value: reflectanceSpectrum)
                return Diffuse(reflectance: reflectanceConstant)
        }
        throw DiffuseError.noReflectance
}
