struct BsdfFrame {

        init() {
                geometricNormal = Normal()
                shadingFrame = ShadingFrame()
        }

        init(geometricNormal: Normal, shadingFrame: ShadingFrame) {
                self.geometricNormal = geometricNormal
                self.shadingFrame = shadingFrame
        }

        init(interaction: any Interaction) {
                let shadingFrame = ShadingFrame(
                        tangent: Vector(normal: interaction.shadingNormal),
                        bitangent: normalized(interaction.dpdu)
                )
                self.geometricNormal = interaction.normal
                self.shadingFrame = shadingFrame
        }

        func isReflecting(wi: Vector, wo: Vector) -> Bool {
                return dot(wi, geometricNormal) * dot(wo, geometricNormal) > 0
        }

        let geometricNormal: Normal
        let shadingFrame: ShadingFrame
}
