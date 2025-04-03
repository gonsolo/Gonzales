enum MeasuredError: Error {
        case emptyFileName
}

struct Measured {

        func getBsdf(interaction: any Interaction) -> some GlobalBsdf {
                // TODO: Implement this some day
                let bsdfFrame = BsdfFrame(interaction: interaction)
                let diffuseBsdf = DiffuseBsdf(bsdfFrame: bsdfFrame)
                return diffuseBsdf
        }
}

func createMeasured(parameters: ParameterDictionary) throws -> Measured {
        return Measured()
}
