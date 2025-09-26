struct BsdfFrame {

        init() {
                geometricNormal = Normal()
                shadingFrame = ShadingFrame()
        }

        init(geometricNormal: Normal, shadingFrame: ShadingFrame) {
                self.geometricNormal = geometricNormal
                self.shadingFrame = shadingFrame
        }

        init(interaction: InteractionType) {
                let shadingFrame = ShadingFrame(
                        x: Vector(normal: interaction.shadingNormal),
                        y: normalized(interaction.dpdu)
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
