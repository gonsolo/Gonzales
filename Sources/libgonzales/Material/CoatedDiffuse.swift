struct CoatedDiffuse {

        init(
                roughness: (FloatX, FloatX),
                reflectance: RgbSpectrumTexture,
                refractiveIndex: FloatTexture,
                remapRoughness: Bool
        ) {
                self.roughness = roughness
                self.reflectance = reflectance
                self.refractiveIndex = refractiveIndex
                self.remapRoughness = remapRoughness
        }

        func getBsdf(interaction: any Interaction) -> CoatedDiffuseBsdf {
                let refractiveIndex = self.refractiveIndex.evaluateFloat(at: interaction)
                let reflectanceAtInteraction = reflectance.evaluateRgbSpectrum(at: interaction)
                let bsdfFrame = BsdfFrame(interaction: interaction)

                let alpha: (FloatX, FloatX) = (0.001, 0.001)
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let dielectric = DielectricBsdf(
                        distribution: distribution, refractiveIndex: refractiveIndex, bsdfFrame: bsdfFrame)
                let diffuse = DiffuseBsdf(reflectance: reflectanceAtInteraction, bsdfFrame: bsdfFrame)

                let coatedDiffuseBsdf = CoatedDiffuseBsdf(
                        dielectric: dielectric,
                        diffuse: diffuse,
                        bsdfFrame: bsdfFrame)
                return coatedDiffuseBsdf
        }

        var reflectance: RgbSpectrumTexture
        var refractiveIndex: FloatTexture
        var roughness: (FloatX, FloatX)
        var remapRoughness: Bool
}

@MainActor
func createCoatedDiffuse(parameters: ParameterDictionary) throws -> CoatedDiffuse {
        let remapRoughness = try parameters.findOneBool(called: "remaproughness", else: true)
        let roughnessOptional = try parameters.findOneFloatXOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "uroughness", else: 0.5)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "vroughness", else: 0.5)
        let roughness = (uRoughness, vRoughness)
        let reflectance = try parameters.findRgbSpectrumTexture(name: "reflectance")
        let refractiveIndex = try parameters.findFloatXTexture(name: "eta", else: 1.5)
        return CoatedDiffuse(
                roughness: roughness,
                reflectance: reflectance,
                refractiveIndex: refractiveIndex,
                remapRoughness: remapRoughness)
}
