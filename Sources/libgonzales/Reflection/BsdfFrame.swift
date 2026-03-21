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
                let ns = faceForward(
                        normal: interaction.shadingNormal,
                        comparedTo: interaction.outgoing)
                let shadingFrame = ShadingFrame(
                        y: normalized(interaction.dpdu),
                        z: Vector(normal: ns)
                )
                self.geometricNormal = faceForward(
                        normal: interaction.normal, comparedTo: interaction.outgoing)
                self.shadingFrame = shadingFrame
        }

        func isReflecting(incident: Vector, outgoing: Vector) -> Bool {
                return dot(incident, geometricNormal) * dot(outgoing, geometricNormal) > 0
        }

        let geometricNormal: Normal
        let shadingFrame: ShadingFrame
}
