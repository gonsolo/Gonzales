final class Dielectric: Material {

        init(
                refractiveIndex: FloatTexture = ConstantTexture(value: FloatX(1)),
                roughness: (FloatX, FloatX),
                remapRoughness: Bool
        ) {
                self.refractiveIndex = refractiveIndex
                self.roughness = roughness
                self.remapRoughness = remapRoughness
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                let refractiveIndex = self.refractiveIndex.evaluateFloat(at: interaction)
                let alpha = remapRoughness ? TrowbridgeReitzDistribution.getAlpha(from: roughness) : roughness
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let bxdf = DielectricBsdf(distribution: distribution, refractiveIndex: refractiveIndex)
                let bsdf = BSDF(bxdf: bxdf, interaction: interaction)
                return bsdf
        }

        let refractiveIndex: FloatTexture
        let roughness: (FloatX, FloatX)
        let remapRoughness: Bool
}

func createDielectric(parameters: ParameterDictionary) throws -> Dielectric {
        let remapRoughness = try parameters.findOneBool(called: "remaproughness", else: true)
        let roughnessOptional = try parameters.findOneFloatXOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "uroughness", else: 0.0)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "vroughness", else: 0.0)
        let roughness = (uRoughness, vRoughness)
        let refractiveIndex = try parameters.findFloatXTexture(name: "eta", else: 1.5)
        return Dielectric(
                refractiveIndex: refractiveIndex,
                roughness: roughness,
                remapRoughness: remapRoughness)
}
