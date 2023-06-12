enum MeasuredError: Error {
        case emptyFileName
}

final class Measured: Material {

        func getGlobalBsdf(interaction: Interaction) -> GlobalBsdf {
                // TODO: Implement this some day
                let bxdf = DiffuseBsdf()
                let bsdf = GlobalBsdf(bxdf: bxdf, interaction: interaction)
                return bsdf
        }
}

func createMeasured(parameters: ParameterDictionary) throws -> Measured {
        return Measured()
}
