enum DiffuseError: Error {
        case noReflectance
        case noTexture
}

struct Diffuse {

        init(reflectance: Texture) {
                self.reflectance = reflectance
        }

        func setBsdf(interaction: inout SurfaceInteraction) {
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
                interaction.bsdf = diffuseBsdf
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
                if let rgbTexture = state.textures[reflectanceTextureName] as? RgbSpectrumTexture {
                        texture = rgbTexture
                }
                return Diffuse(reflectance: texture)
        }
        if let reflectanceSpectrum = try parameters.findSpectrum(name: "reflectance", else: gray)
                as? RgbSpectrum
        {
                let reflectanceConstant = ConstantTexture<RgbSpectrum>(value: reflectanceSpectrum)
                return Diffuse(reflectance: reflectanceConstant)
        }
        throw DiffuseError.noReflectance
}
