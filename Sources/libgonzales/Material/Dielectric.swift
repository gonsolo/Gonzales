struct Dielectric {

        init(
                refractiveIndex: Texture,
                roughness: (Real, Real),
                remapRoughness: Bool
        ) {
                self.refractiveIndex = refractiveIndex
                self.roughness = roughness
                self.remapRoughness = remapRoughness
        }

        func getBsdf(interaction: SurfaceInteraction, arena: TextureArena) -> DielectricBsdf {
                let refractiveIndex = self.refractiveIndex.evaluateFloat(at: interaction, arena: arena)
                let alpha = remapRoughness ? TrowbridgeReitzDistribution.getAlpha(from: roughness) : roughness
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let dielectricBsdf = DielectricBsdf(
                        distribution: distribution,
                        refractiveIndex: refractiveIndex,
                        bsdfFrame: bsdfFrame)
                return dielectricBsdf
        }

        let refractiveIndex: Texture
        let roughness: (Real, Real)
        let remapRoughness: Bool
}

extension Dielectric {
        static func create(parameters: ParameterDictionary, textures: [String: Texture], arena: inout TextureArena) throws -> Dielectric {
        let remapRoughness = try parameters.findOneBool(called: "remaproughness", else: true)
        let roughnessOptional = try parameters.findOneRealOptional(called: "roughness")
        let uRoughness =
                try roughnessOptional ?? parameters.findOneReal(called: "uroughness", else: 0.0)
        let vRoughness =
                try roughnessOptional ?? parameters.findOneReal(called: "vroughness", else: 0.0)
        let roughness = (uRoughness, vRoughness)
        let refractiveIndex = try parameters.findRealTexture(name: "eta", textures: textures, arena: &arena, else: 1.5)
        return Dielectric(
                refractiveIndex: refractiveIndex,
                roughness: roughness,
                remapRoughness: remapRoughness)
}
}
