struct DiffuseTransmission {

        func getBsdf(interaction: SurfaceInteraction, arena: TextureArena) -> DiffuseBsdf {
                let reflectance = reflectance.evaluateRgbSpectrum(at: interaction, arena: arena)
                let scale = scale.evaluateFloat(at: interaction, arena: arena)
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let diffuseBsdf = DiffuseBsdf(
                        reflectance: reflectance * scale,
                        bsdfFrame: bsdfFrame)
                return diffuseBsdf
        }

        var reflectance: Texture
        var transmittance: Texture
        var scale: Texture
}

extension DiffuseTransmission {
        static func create(parameters: ParameterDictionary, textures: [String: Texture], arena: inout TextureArena) throws -> DiffuseTransmission {
        let reflectance = try parameters.findRgbSpectrumTexture(
                name: "reflectance",
                textures: textures, arena: &arena,
                else: RgbSpectrum(intensity: 1))
        let transmittance = try parameters.findRgbSpectrumTexture(
                name: "transmittance",
                textures: textures, arena: &arena,
                else: RgbSpectrum(intensity: 1))
        let scale = try parameters.findRealTexture(name: "scale", textures: textures, arena: &arena, else: 1.0)
        return DiffuseTransmission(
                reflectance: reflectance,
                transmittance: transmittance,
                scale: scale)
}
}
