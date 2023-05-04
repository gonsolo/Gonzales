enum MeasuredError: Error {
        case emptyFileName
}

final class Measured: Material {

        func getBSDF(interaction: Interaction) -> BSDF {
                // TODO: Implement this some day
                var bsdf = BSDF(interaction: interaction)
                bsdf.set(bxdf: DiffuseBsdf())
                return bsdf
        }
}

func createMeasured(parameters: ParameterDictionary) throws -> Measured {
        return Measured()
}
