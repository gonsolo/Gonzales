struct DiffuseTransmission {

        func getBsdf(interaction: any Interaction) -> DiffuseBsdf {
                let reflectance = reflectance.evaluateRgbSpectrum(at: interaction)
                let scale = scale.evaluateFloat(at: interaction)
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let diffuseBsdf = DiffuseBsdf(
                        reflectance: scale * reflectance,
                        bsdfFrame: bsdfFrame)
                return diffuseBsdf
        }

        var reflectance: RgbSpectrumTexture
        var transmittance: RgbSpectrumTexture
        var scale: FloatTexture
}

extension DiffuseTransmission {
        static func create(parameters: ParameterDictionary, textures: [String: Texture]) throws -> DiffuseTransmission {
        let reflectance = try parameters.findRgbSpectrumTexture(
                name: "reflectance",
                textures: textures,
                else: RgbSpectrum(intensity: 1))
        let transmittance = try parameters.findRgbSpectrumTexture(
                name: "transmittance",
                textures: textures,
                else: RgbSpectrum(intensity: 1))
        let scale = try parameters.findRealTexture(name: "scale", textures: textures, else: 1.0)
        return DiffuseTransmission(
                reflectance: reflectance,
                transmittance: transmittance,
                scale: scale)
}
}
