enum DiffuseError: Error {
        case noReflectance
        case noTexture
}

final class Diffuse: Material {

        init(reflectance: Texture) {
                self.reflectance = reflectance
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let evaluation = reflectance.evaluate(at: interaction)
                var reflectance = black
                let reflectanceFloat = evaluation as? FloatX
                if reflectanceFloat != nil {
                        reflectance = RGBSpectrum(intensity: reflectanceFloat!)
                }
                let reflectanceRgb = evaluation as? RGBSpectrum
                if reflectanceRgb != nil {
                        reflectance = reflectanceRgb!
                }
                bsdf.set(bxdf: DiffuseBsdf(reflectance: reflectance))
                return bsdf
        }

        let reflectance: Texture
}

func createDiffuse(parameters: ParameterDictionary) throws -> Diffuse {
        let reflectanceTextureName = try parameters.findTexture(name: "reflectance")
        if !reflectanceTextureName.isEmpty {
                var texture: Texture = ConstantTexture(value: black)
                if let floatTexture = state.textures[reflectanceTextureName] as? FloatTexture {
                        texture = floatTexture
                }
                if let rgbTexture = state.textures[reflectanceTextureName] as? RGBSpectrumTexture {
                        texture = rgbTexture
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
