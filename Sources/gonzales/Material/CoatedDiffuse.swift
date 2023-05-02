final class CoatedDiffuse: Material {

        init(
                roughness: (FloatX, FloatX),
                reflectance: RGBSpectrumTexture,
                remapRoughness: Bool
        ) {
                self.roughness = roughness
                self.reflectance = reflectance
                self.remapRoughness = remapRoughness
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let reflectanceAtInteraction = reflectance.evaluateRGBSpectrum(at: interaction)
                let bxdf = CoatedDiffuseBsdf(
                        reflectance: reflectanceAtInteraction,
                        roughness: roughness,
                        remapRoughness: remapRoughness)
                bsdf.set(bxdf: bxdf)
                return bsdf
        }

        var reflectance: RGBSpectrumTexture
        var roughness: (FloatX, FloatX)
        var remapRoughness: Bool
}

func createCoatedDiffuse(parameters: ParameterDictionary) throws -> CoatedDiffuse {
        let remapRoughness = try parameters.findOneBool(called: "remaproughness", else: true)
        let roughnessOptional = try parameters.findOneFloatXOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "uroughness", else: 0.5)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "vroughness", else: 0.5)
        let roughness = (uRoughness, vRoughness)
        let reflectance = try parameters.findRGBSpectrumTexture(name: "reflectance")
        return CoatedDiffuse(
                roughness: roughness,
                reflectance: reflectance,
                remapRoughness: remapRoughness)
}
