enum DiffuseError: Error {
        case noReflectance
        case noTexture
}

struct Diffuse {

        func getBsdf(interaction: InteractionType) -> DiffuseBsdf {
                let evaluation = reflectance.evaluate(at: interaction)
                var reflectance = black
                let reflectanceFloat = evaluation as? FloatX
                if reflectanceFloat != nil {
                        reflectance = RgbSpectrum(intensity: reflectanceFloat!)
                }
                let reflectanceRgb = evaluation as? RgbSpectrum
                if reflectanceRgb != nil {
                        reflectance = reflectanceRgb!
                }
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let diffuseBsdf = DiffuseBsdf(reflectance: reflectance, bsdfFrame: bsdfFrame)
                return diffuseBsdf
        }

        let reflectance: Texture
}

@MainActor
func createDiffuse(parameters: ParameterDictionary) throws -> Diffuse {
        let reflectanceTextureName = try parameters.findTexture(name: "reflectance")
        if !reflectanceTextureName.isEmpty {
                let texture: Texture =
                        state.textures[reflectanceTextureName]
                        ?? Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: black)))
                return Diffuse(reflectance: texture)
        }
        if let reflectanceSpectrum = try parameters.findSpectrum(name: "reflectance", else: gray)
                as? RgbSpectrum
        {
                let constantTexture = ConstantTexture<RgbSpectrum>(value: reflectanceSpectrum)
                let rgbSpectrumTexture = RgbSpectrumTexture.constantTexture(constantTexture)
                let texture = Texture.rgbSpectrumTexture(rgbSpectrumTexture)
                return Diffuse(reflectance: texture)
        }
        throw DiffuseError.noReflectance
}
