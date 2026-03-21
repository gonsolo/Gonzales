enum DiffuseError: Error {
        case noReflectance
        case noTexture
}

struct Diffuse {

        func getBsdf(interaction: SurfaceInteraction) -> DiffuseBsdf {
                let evaluation = reflectance.evaluate(at: interaction)
                var reflectance = black
                let reflectanceFloat = evaluation as? Real
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

extension Diffuse {
        static func create(parameters: ParameterDictionary, textures: [String: Texture]) throws -> Diffuse {
        let reflectanceTextureName = try parameters.findTexture(name: "reflectance")

        if !reflectanceTextureName.isEmpty {

                let texture: Texture =
                        textures[reflectanceTextureName]
                        ?? Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: black)))
                return Diffuse(reflectance: texture)
        }
        if let reflectanceSpectrum = try parameters.findSpectrum(name: "reflectance", else: gray)
                as? RgbSpectrum {
                let constantTexture = ConstantTexture<RgbSpectrum>(value: reflectanceSpectrum)
                let rgbSpectrumTexture = RgbSpectrumTexture.constantTexture(constantTexture)
                let texture = Texture.rgbSpectrumTexture(rgbSpectrumTexture)
                return Diffuse(reflectance: texture)
        }
        throw DiffuseError.noReflectance
}
}
