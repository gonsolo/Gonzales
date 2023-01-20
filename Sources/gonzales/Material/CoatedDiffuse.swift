final class CoatedDiffuse: Material {

        init(roughness: (FloatX, FloatX), reflectance: RGBSpectrumTexture) {
                self.roughness = roughness
                self.reflectance = reflectance
        }

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let reflectanceAtInteraction = reflectance.evaluateRGBSpectrum(at: interaction)
                bsdf.set(bxdf: LambertianReflection(reflectance: reflectanceAtInteraction))
                return bsdf
        }

        var roughness: (FloatX, FloatX)
        var reflectance: RGBSpectrumTexture
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
