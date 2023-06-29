enum MeasuredError: Error {
        case emptyFileName
}

final class Measured {

        func setBsdf(interaction: inout SurfaceInteraction) {
                // TODO: Implement this some day
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let diffuseBsdf = DiffuseBsdf(bsdfFrame: bsdfFrame)
                interaction.bsdf = diffuseBsdf
        }
}

func createMeasured(parameters: ParameterDictionary) throws -> Measured {
        return Measured()
}
