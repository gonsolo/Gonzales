final class Dielectric: Material {

        init(
                refractiveIndex: FloatTexture = ConstantTexture(value: FloatX(1)),
                roughness: (FloatX, FloatX)
        ) {
                self.refractiveIndex = refractiveIndex
                self.roughness = roughness
        }

        func getBSDF(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                let refractiveIndex = self.refractiveIndex.evaluateFloat(at: interaction)
                let alpha = TrowbridgeReitzDistribution.getAlpha(from: roughness)
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let bxdf = DielectricBsdf(distribution: distribution, refractiveIndex: refractiveIndex)
                bsdf.set(bxdf: bxdf)
                return bsdf
        }

        let refractiveIndex: FloatTexture
        let roughness: (FloatX, FloatX)
}

func createDielectric(parameters: ParameterDictionary) throws -> Dielectric {
        let roughnessOptional = try parameters.findOneFloatXOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "uroughness", else: 0.0)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneFloatX(called: "vroughness", else: 0.0)
        let roughness = (uRoughness, vRoughness)
        let refractiveIndex = try parameters.findFloatXTexture(name: "eta", else: 1.5)
        return Dielectric(refractiveIndex: refractiveIndex, roughness: roughness)
}
