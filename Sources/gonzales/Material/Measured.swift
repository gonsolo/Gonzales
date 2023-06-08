enum MeasuredError: Error {
        case emptyFileName
}

final class Measured: Material {

        func getBSDF(interaction: Interaction) -> BSDF {
                // TODO: Implement this some day
                let bxdf = DiffuseBsdf()
                let bsdf = BSDF(bxdf: bxdf, interaction: interaction)
                return bsdf
        }
}

func createMeasured(parameters: ParameterDictionary) throws -> Measured {
        return Measured()
}
