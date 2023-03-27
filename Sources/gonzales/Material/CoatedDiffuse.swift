final class CoatedDiffuse: Material {

        init(roughness: (FloatX, FloatX), reflectance: RGBSpectrumTexture) {
                self.roughness = roughness
                self.reflectance = reflectance
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let reflectanceAtInteraction = reflectance.evaluateRGBSpectrum(at: interaction)
                let bxdf = CoatedDiffuseBsdf(
                        reflectance: reflectanceAtInteraction,
                        roughness: roughness)
                bsdf.set(bxdf: bxdf)
                return bsdf
        }

        var reflectance: RGBSpectrumTexture
        var roughness: (FloatX, FloatX)
}

func createCoatedDiffuse(parameters: ParameterDictionary) throws -> CoatedDiffuse {
        // remaproughness assumed to be false
        let roughnessOptional = try parameters.findOneFloatXOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "uroughness", else: 0.5)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "vroughness", else: 0.5)
        let roughness = (uRoughness, vRoughness)
        let reflectance = try parameters.findRGBSpectrumTexture(name: "reflectance")
        return CoatedDiffuse(roughness: roughness, reflectance: reflectance)
}
