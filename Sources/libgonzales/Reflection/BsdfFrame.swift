public struct BsdfFrame: Sendable {

        init() {
                geometricNormal = Normal()
                shadingFrame = ShadingFrame()
        }

        public init(geometricNormal: Normal, shadingFrame: ShadingFrame) {
                self.geometricNormal = geometricNormal
                self.shadingFrame = shadingFrame
        }

        init(interaction: any Interaction) {
                let shadingFrame = ShadingFrame(
                        y: normalized(interaction.dpdu),
                        z: Vector(normal: interaction.shadingNormal)
                )
                self.geometricNormal = interaction.normal
                self.shadingFrame = shadingFrame
        }

        func isReflecting(incident: Vector, outgoing: Vector) -> Bool {
                return dot(incident, geometricNormal) * dot(outgoing, geometricNormal) > 0
        }

        let geometricNormal: Normal
        let shadingFrame: ShadingFrame
}
