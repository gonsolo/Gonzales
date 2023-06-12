struct BsdfGeometry {

        init() {
                geometricNormal = Normal()
                frame = ShadingFrame()
        }

        init(geometricNormal: Normal, frame: ShadingFrame) {
                self.geometricNormal = geometricNormal
                self.frame = frame
        }

        init(interaction: Interaction) {
                let frame = ShadingFrame(
                        x: Vector(normal: interaction.shadingNormal),
                        y: normalized(interaction.dpdu)
                )
                self.geometricNormal = interaction.normal
                self.frame = frame
        }

        func isReflecting(wi: Vector, wo: Vector) -> Bool {
                return dot(wi, geometricNormal) * dot(wo, geometricNormal) > 0
        }

        let geometricNormal: Normal
        let frame: ShadingFrame
}
