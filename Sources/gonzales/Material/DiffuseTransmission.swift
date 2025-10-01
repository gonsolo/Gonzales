struct DiffuseTransmission {

        func getBsdf(interaction: any Interaction) -> DiffuseBsdf {
                let reflectance = reflectance.evaluateRgbSpectrum(at: interaction)
                let scale = scale.evaluateFloat(at: interaction)
                // TODO: check same hemisphere and transmission
                // TODO: transmittance
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

@MainActor
func createDiffuseTransmission(parameters: ParameterDictionary) throws -> DiffuseTransmission {
        let reflectance = try parameters.findRgbSpectrumTexture(
                name: "reflectance",
                else: RgbSpectrum(intensity: 1))
        let transmittance = try parameters.findRgbSpectrumTexture(
                name: "transmittance",
                else: RgbSpectrum(intensity: 1))
        let scale = try parameters.findFloatXTexture(name: "scale", else: 1.0)
        return DiffuseTransmission(
                reflectance: reflectance,
                transmittance: transmittance,
                scale: scale)
}
